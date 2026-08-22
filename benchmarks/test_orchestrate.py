import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import unittest.mock
from pathlib import Path

import fixture_digest
import orchestrate
from runners import bench_common

HERE = Path(__file__).resolve().parent


def result(framework, backend, variant, losses, precision="f32"):
    return {
        "workload": "mlp_wide",
        "framework": framework,
        "backend": backend,
        "variant": variant,
        "precision": precision,
        "losses": losses,
    }


def cell(framework, backend, variant, losses, precision="f32", p50=1.0):
    """A result with the fields `report` reads, not just the parity ones."""
    r = result(framework, backend, variant, losses, precision=precision)
    r.update(
        {
            "step_ms": {"p10": p50, "p50": p50, "p90": p50},
            "queued_step_ms": p50,
            "compile_s": 1.5,
        }
    )
    return r


def _reject(token):
    raise ValueError(f"not JSON: {token}")


def strict_loads(text):
    """`json.loads` without its NaN/Infinity extension -- JSON as every other reader has it."""
    return json.loads(text, parse_constant=_reject)


class ParityCheckTest(unittest.TestCase):
    def test_precision_tolerances_match_measured_headroom(self):
        self.assertEqual(orchestrate.PARITY_TOL_PRECISION, {"bf16": 4e-3, "f16": 2e-3})

    def test_flat_candidate_fails_even_when_within_bf16_tolerance(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        flat = result(
            "ocannl", "metal", "default", [2.3026, 2.3026, 2.3026], precision="bf16"
        )

        orchestrate.parity_check([ref, flat])

        self.assertLess(flat["parity_max_rel"], orchestrate.PARITY_TOL_PRECISION["bf16"])
        self.assertFalse(flat["parity_loss_moved"])
        self.assertEqual(flat["parity"], "FAIL")

    def test_moving_candidate_within_tolerance_passes(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        moving = result(
            "ocannl", "metal", "tuned", [2.3025, 2.3012, 2.3004], precision="bf16"
        )

        orchestrate.parity_check([ref, moving])

        self.assertTrue(moving["parity_loss_moved"])
        self.assertEqual(moving["parity"], "PASS")


class DivergenceTest(unittest.TestCase):
    """gh-ocannl-676: a cell whose training diverged ran, and its trajectory is the finding.

    The three runners now all emit `null` for a number JSON cannot express, and the OCANNL one
    used to emit OCaml's `nan`, whose line `json.loads` refuses -- so the cell was reported as a
    broken runner and the loss vector that shows the divergence was thrown away. Here the verdict
    is DIVERGED: a parity failure that names its cause.
    """

    def test_null_loss_is_a_diverged_cell_not_a_missing_one(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("ocannl", "metal", "default", [2.3026, None, None], precision="f16")

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)
        self.assertEqual(ref["parity"], "REF")

    def test_nan_from_a_python_runner_reads_the_same(self):
        # Python's json.loads accepts its own `NaN` extension, so an older result file can still
        # deliver one; it is the same fact as a null.
        ref = result("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = result("tinygrad", "metal", "jit", [2.3026, float("nan"), float("inf")])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)

    def test_a_diverged_cell_is_not_reported_as_stationary(self):
        # Its finite prefix moved; and with fewer than two finite steps there is nothing to say
        # about movement, which is reported as divergence rather than as a flat trajectory.
        self.assertTrue(orchestrate.loss_moved([2.3026, 1.5, None]))
        self.assertFalse(orchestrate.loss_moved([2.3026, None, None]))

    def test_the_finite_prefix_is_still_compared(self):
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, None])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)

    def test_a_finite_value_after_the_divergence_is_not_evidence(self):
        # A loss that comes back finite after a NaN is whatever the arithmetic settled on, not
        # drift: comparing it would put a number on the DIVERGED row that its own contract calls
        # meaningless, and reading it as movement would say the trajectory moved when what it did
        # was blow up.
        ref = result("pytorch", "cpu", "eager", [2.0, 2.0, 2.0])
        blown = result("ocannl", "cc", "default", [2.0, None, 9.0])

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity_max_rel"], 0.0)
        self.assertFalse(blown["parity_loss_moved"])
        self.assertEqual(orchestrate.finite_prefix([2.0, None, 9.0]), [2.0])

    def test_the_reference_prefix_bounds_the_comparison_too(self):
        ref = result("pytorch", "cpu", "eager", [2.0, float("nan"), 2.0])
        other = result("ocannl", "cc", "default", [2.0, 5.0, 5.0])

        orchestrate.parity_check([ref, other])

        self.assertEqual(other["parity_max_rel"], 0.0)

    def test_a_diverged_reference_is_no_reference(self):
        ref = result("pytorch", "cpu", "eager", [2.3026, None, None])
        other = result("ocannl", "cc", "default", [2.3026, 2.3010, 2.3000])

        orchestrate.parity_check([ref, other])

        self.assertEqual(ref["parity"], "DIVERGED")
        self.assertEqual(other["parity"], "NO-REF")

    def test_report_names_the_divergence_and_writes_parseable_json(self):
        ref = cell("pytorch", "cpu", "eager", [2.3026, 2.3010, 2.3000])
        blown = cell(
            "ocannl", "metal", "default", [2.3026, None, None], precision="f16", p50=None
        )
        orchestrate.parity_check([ref, blown])

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                orchestrate.report([ref, blown], out)
            text = (out / "report.md").read_text()
            rows = [
                strict_loads(line) for line in (out / "results.jsonl").read_text().splitlines()
            ]

        self.assertIn("DIVERGED", text)
        self.assertIn("loss non-finite from step 1", text)
        # A time the runner never measured prints as n/a rather than crashing the report.
        self.assertIn("n/a", text)
        self.assertEqual([r["parity"] for r in rows], ["REF", "DIVERGED"])

    def test_emit_writes_null_for_a_non_finite_number(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bench_common.emit({"losses": [1.0, float("nan")], "step_ms": {"p50": float("inf")}})

        parsed = strict_loads(buf.getvalue().strip())
        self.assertEqual(parsed, {"losses": [1.0, None], "step_ms": {"p50": None}})


class OcannlResultLineTest(unittest.TestCase):
    """The OCANNL runner's own result line, as this orchestrator reads it (gh-ocannl-676).

    The golden of test/operations/bench_result_line holds the line built from fabricated values --
    a diverged trajectory, times that were never measured, a diagnostic full of control characters
    -- and this is the parser that has to accept it. Pinning it in OCaml alone would leave the
    claim "this parses" resting on a second implementation of JSON.
    """

    GOLDEN = HERE.parent / "test/operations/bench_result_line.expected"

    def lines(self):
        return [
            line
            for line in self.GOLDEN.read_text().splitlines()
            if line.startswith("{")  # what run_cell picks out of a cell's output
        ]

    def test_every_emitted_line_parses_strictly(self):
        lines = self.lines()
        self.assertGreaterEqual(len(lines), 2, self.GOLDEN)
        for line in lines:
            with self.subTest(line=line[:60]):
                self.assertIn("losses", strict_loads(line))

    def test_the_diverged_line_is_read_as_a_diverged_cell(self):
        parsed = [strict_loads(line) for line in self.lines()]
        blown = next(r for r in parsed if None in r["losses"])
        ref = result("pytorch", "cpu", "eager", [2.5, 1.75, 1.25])
        ref["workload"] = blown["workload"]

        orchestrate.parity_check([ref, blown])

        self.assertEqual(blown["parity"], "DIVERGED")
        self.assertEqual(blown["diverged_at"], 1)


class CellIdentityTest(unittest.TestCase):
    """gh-ocannl-539: scheduling variant and storage precision are independent axes."""

    def test_f32_cells_keep_their_bare_variant_name(self):
        self.assertEqual(orchestrate.cell_name("tuned", "f32"), "tuned")
        self.assertEqual(orchestrate.cell_name("default", "f32"), "default")

    def test_reduced_precision_cells_are_named_by_the_product(self):
        self.assertEqual(orchestrate.cell_name("tuned", "bf16"), "tuned/bf16")
        self.assertEqual(orchestrate.cell_name("default", "f16"), "default/f16")

    def test_skip_entries_are_precision_wildcards(self):
        # A None precision means "at every precision": a scheduling pathology must not be let back
        # in through the bf16/f16 columns. Tested against an injected entry rather than a live one,
        # so emptying SKIP_CELLS does not silently drop coverage of the wildcard itself.
        entry = ("some_workload", "metal", "tuned", None)
        orchestrate.SKIP_CELLS.add(entry)
        try:
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "f32"))
            self.assertTrue(orchestrate.cell_skipped("some_workload", "metal", "tuned", "bf16"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "metal", "default", "f32"))
            self.assertFalse(orchestrate.cell_skipped("some_workload", "cc", "tuned", "bf16"))
        finally:
            orchestrate.SKIP_CELLS.discard(entry)

    def test_cifar_conv_metal_tuned_is_no_longer_skipped(self):
        # gh-ocannl-538: the post-tune re-init hang the entry was added for did not reproduce —
        # the cell completed in ~4 min with no hang, and the sweep reproduced its p50 to 0.005%.
        for precision in ("f32", "bf16"):
            self.assertFalse(
                orchestrate.cell_skipped("cifar_conv", "metal", "tuned", precision), precision
            )

    def test_hip_tuned_cells_are_no_longer_skipped(self):
        # gh-ocannl-532's unparallelized-baseline dispatch is fixed in the tuner and confirmed on
        # the gfx1151 machine that produced the symptom; all three cells complete.
        for workload in ("mlp_wide", "cifar_conv", "cifar_stride"):
            for precision in ("f32", "bf16"):
                self.assertFalse(
                    orchestrate.cell_skipped(workload, "hip", "tuned", precision),
                    (workload, precision),
                )

    def test_every_skip_entry_is_a_four_tuple(self):
        for entry in orchestrate.SKIP_CELLS:
            self.assertEqual(len(entry), 4, entry)

    def test_tuned_f32_and_tuned_bf16_are_distinct_cells(self):
        rows = [
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1]),
            result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1], precision="bf16"),
        ]
        names = {orchestrate.cell_name(r["variant"], r["precision"]) for r in rows}

        self.assertEqual(names, {"tuned", "tuned/bf16"})


class ReportGroupingTest(unittest.TestCase):
    def test_rows_group_by_precision_f32_first(self):
        self.assertEqual(orchestrate.precision_rank("f32"), 0)
        self.assertLess(
            orchestrate.precision_rank("bf16"), orchestrate.precision_rank("f16")
        )
        # A gate leg sorts directly after the storage precision it varies, not last: it is a
        # variant of how f16's optimizer step is gated, and reads next to plain f16.
        self.assertGreater(
            orchestrate.precision_rank("f16-static"), orchestrate.precision_rank("f16")
        )
        self.assertLess(
            orchestrate.precision_rank("f16-gated8"),
            orchestrate.precision_rank("unknown-precision"),
        )

    def test_gate_legs_are_parity_gated_at_their_base_precision(self):
        self.assertEqual(
            orchestrate.parity_tol("f16-gated8"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )
        self.assertEqual(
            orchestrate.parity_tol("f16-static"), orchestrate.PARITY_TOL_PRECISION["f16"]
        )


class PrecisionLegTest(unittest.TestCase):
    """gh-ocannl-551: the gate-cost legs are orchestrated cells, and an inexpressible cell is
    reported rather than silently absent."""

    def test_gate_legs_dispatch_their_own_flags(self):
        self.assertEqual(
            orchestrate.precision_env("f16-static"),
            {"BENCH_PRECISION": "f16", "BENCH_STATIC_SCALE": "1"},
        )
        self.assertEqual(
            orchestrate.precision_env("f16-gated16"),
            {"BENCH_PRECISION": "f16", "BENCH_GATE_INTERVAL": "16"},
        )
        self.assertEqual(orchestrate.precision_env("bf16"), {"BENCH_PRECISION": "bf16"})

    def test_cell_env_carries_the_leg_and_clears_the_others(self):
        # The gate flags are cleared per cell and then set by the leg — the two collided as
        # duplicate dict() keywords, which crashed the run only once a gate cell was dispatched.
        stray = {"BENCH_STATIC_SCALE": "1", "BENCH_GATE_INTERVAL": "7"}
        plain = orchestrate.cell_env(stray, "fx.safetensors", "default", "bf16")
        self.assertEqual(plain["BENCH_PRECISION"], "bf16")
        self.assertEqual(plain["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(plain["BENCH_GATE_INTERVAL"], "0")
        gated = orchestrate.cell_env(stray, "fx.safetensors", "tuned", "f16-gated16")
        self.assertEqual(gated["BENCH_PRECISION"], "f16")
        self.assertEqual(gated["BENCH_GATE_INTERVAL"], "16")
        self.assertEqual(gated["BENCH_STATIC_SCALE"], "0")
        self.assertEqual(gated["BENCH_TUNE"], "1")
        static = orchestrate.cell_env(stray, "fx.safetensors", "default", "f16-static")
        self.assertEqual(static["BENCH_STATIC_SCALE"], "1")
        self.assertEqual(static["BENCH_GATE_INTERVAL"], "0")

    def test_precision_spec_rejects_nonsense(self):
        for spec in ("f16-gated0", "f16-gated", "f8", "f16-dynamic"):
            with self.assertRaises(Exception, msg=spec):
                orchestrate.precision_spec(spec)

    def test_gate_legs_need_a_training_workload(self):
        # The forward-only gpt2_mini has no optimizer, hence no loss scale to gate.
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-static")
        )
        self.assertIsNotNone(
            orchestrate.precision_unavailable("gpt", "infer", "f16-gated8")
        )
        # ... but gpt2_mini_train does, and plain reduced precisions work in both modes.
        self.assertIsNone(
            orchestrate.precision_unavailable("gpt", "train", "f16-static")
        )
        self.assertIsNone(orchestrate.precision_unavailable("gpt", "infer", "bf16"))
        self.assertIsNone(orchestrate.precision_unavailable("mlp", "train", "f16-gated8"))

    def test_conv_workloads_report_the_missing_runner_support(self):
        reason = orchestrate.precision_unavailable("conv", "train", "bf16")
        self.assertIsNotNone(reason)
        self.assertIn("conv", reason)


class ProvenanceTest(unittest.TestCase):
    """gh-ocannl-644: a tuned cell's result says which pass timed it."""

    def tuned(self, searched, searches=0, replays=0, no_searches=0):
        r = result("ocannl", "hip", "tuned", [2.3, 2.2, 2.1])
        if searched is not None:
            r["searched"] = searched
            r["tune"] = {
                "shipped": "A", "searches": searches, "replays": replays,
                "no_searches": no_searches, "arms": []
            }
        return r

    def test_a_replay_is_the_compliant_case(self):
        replay = self.tuned(False, replays=2)

        self.assertEqual(orchestrate.provenance_check([replay]), [])
        self.assertEqual(replay["provenance"], "REPLAY")

    def test_search_pass_timings_are_a_violation(self):
        # The failure this exists to catch: both passes emit the same framework/backend/variant/
        # precision, so before `searched` a report could quote pass-1 numbers indefinitely.
        searching = self.tuned(True, searches=2)

        self.assertEqual(orchestrate.provenance_check([searching]), [searching])
        self.assertEqual(searching["provenance"], "SEARCH-PASS")

    def test_one_classifier_reads_searched_for_every_consumer(self):
        # The round-1 and round-2 findings were the same shape — a three-valued fact read through
        # a boolean, wrong in a different place each time. There is now one reading of it, and the
        # gate, the `pass` column and the carried-over compile_s label all go through it.
        tune = lambda searches, replays: {  # noqa: E731
            "shipped": "A", "searches": searches, "replays": replays, "arms": []
        }
        searched = {"searched": True, "tune": tune(2, 0)}
        replay = {"searched": False, "tune": tune(0, 2)}
        no_search = {"searched": False, "tune": tune(0, 0)}

        self.assertEqual(orchestrate.search_provenance(searched), "SEARCHED")
        self.assertEqual(orchestrate.search_provenance(replay), "REPLAY")
        self.assertEqual(orchestrate.search_provenance(no_search), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance({"searched": None}), "UNKNOWN")
        # No tune object: `searched: false` alone cannot separate a replay from a cell that
        # searched nothing, and guessing either way has already been a bug.
        self.assertEqual(orchestrate.search_provenance({"searched": False}), "UNKNOWN")

    def test_the_runner_states_the_no_search_case_instead_of_it_being_derived(self):
        # gh-ocannl-677: the OCaml call's outcome is one state, and the runner now names it per
        # arm and counts it (`no_searches`) rather than leaving the reader to recover "neither
        # searched nor replayed" from two zeroed counters in a JSON artifact.
        stated = lambda replays, no_searches: {  # noqa: E731
            "searched": False,
            "tune": {"shipped": "A", "searches": 0, "replays": replays,
                     "no_searches": no_searches, "arms": []},
        }

        self.assertEqual(orchestrate.search_provenance(stated(0, 2)), "NO-SEARCH")
        self.assertEqual(orchestrate.search_provenance(stated(2, 0)), "REPLAY")
        # A mixed cell — one arm replayed, the other had nothing to replay — still carries a tuned
        # artifact, so it is a replay.
        self.assertEqual(orchestrate.search_provenance(stated(1, 1)), "REPLAY")

    def test_only_a_real_replay_labels_the_carried_over_compile_cost_cached(self):
        # A search pass under autotune_search=false hands over the cost of compiling the untuned
        # default; calling that "(cached)" would claim a schedule cache it never touched.
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("REPLAY"), " (cached)")
        self.assertEqual(orchestrate.COMPILE_S_NOTE.get("NO-SEARCH"), " (no search)")
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("SEARCHED"))
        self.assertIsNone(orchestrate.COMPILE_S_NOTE.get("UNKNOWN"))

    def test_a_disabled_search_is_neither_a_violation_nor_a_replay(self):
        # gh-ocannl-559's reproducible profile turns the search off: the cell ships the untuned
        # default, having neither searched nor replayed. Gating it would fail BOTH passes of every
        # tuned cell, and calling it a replay would credit the row with a tuned artifact it does
        # not have.
        cell = self.tuned(False, searches=0, replays=0, no_searches=2)

        self.assertEqual(orchestrate.provenance_check([cell]), [])
        self.assertEqual(cell["provenance"], "NO-SEARCH")

        # And the same verdict from an artifact written before `no_searches` existed.
        legacy = self.tuned(False, searches=0, replays=0)
        del legacy["tune"]["no_searches"]

        self.assertEqual(orchestrate.provenance_check([legacy]), [])
        self.assertEqual(legacy["provenance"], "NO-SEARCH")

    def test_a_runner_without_the_field_is_unknown_not_a_violation(self):
        old = self.tuned(None)

        self.assertEqual(orchestrate.provenance_check([old]), [])
        self.assertEqual(old["provenance"], "UNKNOWN")

    def test_cells_that_neither_search_nor_compile_are_not_annotated(self):
        # They have no process distinction to be on the wrong side of; a `pass` entry for them
        # would be a claim about a protocol they do not run.
        default = result("ocannl", "hip", "default", [2.3, 2.2, 2.1])
        default["searched"] = False
        eager = result("pytorch", "cpu", "eager", [2.3, 2.2, 2.1])
        jit = result("tinygrad", "CPU", "jit", [2.3, 2.2, 2.1])

        self.assertEqual(orchestrate.provenance_check([default, eager, jit]), [])
        for r in (default, eager, jit):
            self.assertNotIn("provenance", r)

    def test_a_single_pass_framework_states_its_search_without_being_gated(self):
        # gh-ocannl-675: tinygrad's beam search and torch.compile run in the timing process by
        # protocol. Nothing yet says they pay for it, so the report states it and the gate keeps
        # its hands off — the alternative, saying nothing, reads as though only OCANNL searches.
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = True
        compiled = result("pytorch", "cuda", "compiled", [2.3, 2.2, 2.1])
        compiled["searched"] = False

        self.assertEqual(orchestrate.provenance_check([beam, compiled]), [])
        self.assertEqual(beam["provenance"], "SAME-PROCESS")
        self.assertEqual(compiled["provenance"], "CACHED")

    def test_a_single_pass_cell_that_cannot_tell_is_unknown(self):
        beam = result("tinygrad", "AMD", "beam", [2.3, 2.2, 2.1])
        beam["searched"] = None

        self.assertEqual(orchestrate.provenance_check([beam]), [])
        self.assertEqual(beam["provenance"], "UNKNOWN")

    def test_every_verdict_renders(self):
        for verdict in (
            "REPLAY",
            "SEARCH-PASS",
            "NO-SEARCH",
            "SAME-PROCESS",
            "CACHED",
            "UNKNOWN",
        ):
            self.assertIn(verdict, orchestrate.PROVENANCE_MARK)

    def test_the_gated_and_stated_cell_sets_do_not_overlap(self):
        self.assertFalse(orchestrate.TWO_PASS_CELLS & orchestrate.SAME_PROCESS_CELLS)


class TensorizationTest(unittest.TestCase):
    """gh-ocannl-626: a "tensorized" timing must not be able to be a scalar-fallback timing."""

    def cell(self, arms, shipped="A", shipped_mma=...):
        r = result("ocannl", "metal", "tuned", [2.3, 2.2, 2.1])
        r["searched"] = False
        r["step_ms"] = {"p10": 1.0, "p50": 1.1, "p90": 1.2}
        r["queued_step_ms"] = 0.1
        r["compile_s"] = 3.0
        r["parity"] = "REF"
        r["parity_loss_moved"] = True
        r["diverged_at"] = None
        r["tune"] = {
            "shipped": shipped, "searches": 0, "replays": 1, "no_searches": 0, "arms": arms
        }
        if shipped_mma is not ...:
            r["tune"]["shipped_mma"] = shipped_mma
        return r

    def mma(self, tensorization, statements=0, scalar_fallbacks=0):
        return {
            "tensorization": tensorization, "statements": statements,
            "scalar_fallbacks": scalar_fallbacks,
        }


    def arm(self, name, tensorized, tensorization, statements=0, fallbacks=0):
        return {
            "arm": name, "state": "cache-replay", "searched": False, "cache_hit": True,
            "best_ms": 1.0, "best_label": "L", "tensorized": tensorized,
            "tensorization": tensorization, "mma_statements": statements,
            "mma_scalar_fallbacks": fallbacks, "mma_seeded": 0, "mma_timed": 0,
            "mma_best_ms": 1.0, "terminal_failure": None,
        }

    def test_a_declined_tensorize_is_not_reported_as_a_tensorized_timing(self):
        # The defect: the schedule asked for tensor cores, every Tile_mma rendered the lane-0
        # scalar loop, and the row still carried a tensorized variant name.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_tensorize_that_emitted_no_statement_is_its_own_verdict(self):
        # Distinct from the fallback: nothing declined, because nothing was emitted to decline.
        cell = self.cell([self.arm("A", True, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_an_honestly_tensorized_cell_is_not_flagged(self):
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "TENSORIZED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_that_never_asked_is_not_a_mismatch(self):
        cell = self.cell([self.arm("A", False, "not-requested")])

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-REQUESTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_a_missing_census_never_reads_as_tensorized(self):
        # The negative control. A runner predating the field, or an arm with no crowned candidate,
        # carries a null label — which must answer UNKNOWN, never the passing reading.
        older = self.cell([{"arm": "A", "best_ms": 1.0}])
        no_winner = self.cell([self.arm("A", False, None)])

        self.assertEqual(orchestrate.tensorization_verdict(older), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_verdict(no_winner), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([older, no_winner]), [])
        # And a cell that tuned nothing at all has no arm to consult, so it gets no column entry
        # rather than a fabricated one.
        self.assertIsNone(orchestrate.tensorization_verdict(result("torch", "cuda", "eager", [1.0])))

    def test_the_shipped_routines_census_outranks_the_arm_reports(self):
        # A gh-555 flip refinement that beats the A/B winner ships under `shipped: "flip"` and is
        # not an arm at all; on the timing_ctx path the tuner can also fall back to the untuned
        # default after the arm was crowned. Either way the arm describes a discarded schedule, so
        # the routine that was actually timed is what the column must report.
        flip = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped="flip",
            shipped_mma=self.mma("scalar-fallback", statements=3, scalar_fallbacks=3),
        )

        self.assertIsNone(orchestrate.shipped_arm(flip))
        self.assertEqual(orchestrate.tensorization_verdict(flip), "SCALAR-FALLBACK")
        self.assertEqual(orchestrate.tensorization_check([flip]), [flip])

    def test_a_fallback_to_the_untuned_default_is_not_reported_as_the_crowned_arm(self):
        # The arm was crowned in the scratch context and its replay was rejected in the production
        # one, so the timed routine has no mma at all while the arm still says tensorized.
        cell = self.cell(
            [self.arm("A", True, "tensorized", statements=4)],
            shipped_mma=self.mma("not-requested"),
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "NOT-EMITTED")
        self.assertEqual(orchestrate.tensorization_check([cell]), [cell])

    def test_a_harness_that_recorded_no_shipped_census_reads_unknown(self):
        # Present-but-empty is a runner that reported arms and forgot the routine: not a finding,
        # and not the passing reading either.
        cell = self.cell([self.arm("A", True, "tensorized", statements=4)], shipped_mma=None)

        self.assertEqual(orchestrate.tensorization_verdict(cell), "UNKNOWN")
        self.assertEqual(orchestrate.tensorization_check([cell]), [])

    def test_an_artifact_predating_shipped_mma_still_reads_its_arm(self):
        # The key absent (not null) is an older result line; the arm is right whenever the crowned
        # candidate WAS the shipped artifact, which is the common case.
        cell = self.cell([self.arm("A", True, "scalar-fallback", statements=2, fallbacks=2)])

        self.assertNotIn("shipped_mma", cell["tune"])
        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_the_verdict_describes_the_arm_that_shipped_not_the_fastest_one(self):
        # tune_placements searches several arms and keeps one artifact; reading any other arm
        # describes a schedule that was discarded (gh-ocannl-638).
        cell = self.cell(
            [
                self.arm("A", True, "tensorized", statements=4),
                self.arm("B", True, "scalar-fallback", statements=2, fallbacks=2),
            ],
            shipped="B",
        )

        self.assertEqual(orchestrate.tensorization_verdict(cell), "SCALAR-FALLBACK")

    def test_every_verdict_renders_in_the_table(self):
        # The same completeness check the provenance column carries: a verdict with no mark would
        # print as an em dash, which is the "never asked" reading.
        for verdict in ("TENSORIZED", "SCALAR-FALLBACK", "NOT-EMITTED", "NOT-REQUESTED", "UNKNOWN"):
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
        for verdict in orchestrate.TENSORIZATION_MISMATCH:
            self.assertIn(verdict, orchestrate.TENSORIZATION_MARK)
            self.assertIn("**", orchestrate.TENSORIZATION_MARK[verdict])

    def test_the_mismatch_is_visible_in_the_rendered_table(self):
        cells = [
            self.cell([self.arm("A", True, "scalar-fallback", statements=3, fallbacks=3)]),
            self.cell([self.arm("A", True, "tensorized", statements=4)]),
        ]
        orchestrate.tensorization_check(cells)
        out = Path(tempfile.mkdtemp())
        orchestrate.report(cells, out)
        table = (out / "report.md").read_text()

        self.assertIn(" mma |", table)
        self.assertIn("**SCALAR FALLBACK**", table)
        self.assertIn("tensorized", table)


class RunnerProvenanceProbeTest(unittest.TestCase):
    """gh-ocannl-644: what the Python runners report, and what they refuse to claim."""

    def test_a_cell_that_runs_no_search_says_so(self):
        self.assertIs(bench_common.tinygrad_searched(None, beam=0), False)
        self.assertIs(bench_common.torch_searched(object(), compiled=False), False)

    def test_beam_counts_distinguish_a_search_from_a_kernel_cache_replay(self):
        self.assertIs(
            bench_common.tinygrad_searched({"call": 4, "hit": 3, "put": 1}, beam=4), True
        )
        self.assertIs(
            bench_common.tinygrad_searched({"call": 3, "hit": 3, "put": 0}, beam=4), False
        )

    def test_a_probe_that_saw_nothing_says_unknown_rather_than_no(self):
        # The internals moved under the probe. A wrong False is exactly the silent claim the
        # field exists to prevent, so this must not read as "replayed a cache".
        self.assertIsNone(bench_common.tinygrad_searched(None, beam=4))
        self.assertIsNone(
            bench_common.tinygrad_searched({"call": 0, "hit": 0, "put": 0}, beam=4)
        )

    def test_torch_reads_the_fx_graph_cache_counters(self):
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        self.assertIs(bench_common.torch_searched(Torch(fxgraph_cache_miss=2), True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_hit=2), compiled=True), False
        )
        self.assertIsNone(bench_common.torch_searched(Torch(), compiled=True))
        self.assertIsNone(bench_common.torch_searched(object(), compiled=True))

    def test_a_beam_search_that_writes_no_cache_entry_still_counts_as_a_search(self):
        # CACHELEVEL=0 / IGNORE_BEAM_CACHE: the search runs and touches no cache. Counting cache
        # traffic alone cannot see it, which is why the probe counts calls too.
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 0, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 1, "put": 0}, beam=4), True)
        self.assertIs(bench_common.tinygrad_searched({"call": 4, "hit": 4, "put": 0}, beam=4), False)

    def test_a_bypassed_torch_graph_is_codegen_not_a_replay(self):
        # A run with more than one graph is mixed: one served from the cache, one the cache
        # refused, is still a graph compiled in the timing process.
        class Torch:
            def __init__(self, **counters):
                self._dynamo = type(
                    "d", (), {"utils": type("u", (), {"counters": {"inductor": counters}})}
                )

        mixed = Torch(fxgraph_cache_hit=3, fxgraph_cache_bypass=1)
        self.assertIs(bench_common.torch_searched(mixed, compiled=True), True)
        self.assertIs(
            bench_common.torch_searched(Torch(fxgraph_cache_bypass=2), compiled=True), True
        )

    def test_the_beam_instrument_counts_only_beam_entries_and_passes_values_through(self):
        stored = {("beam_search", "k1"): "winner"}
        calls = []

        class Search:
            @staticmethod
            def diskcache_get(table, key):
                calls.append(("get", table, key))
                return stored.get((table, key))

            @staticmethod
            def diskcache_put(table, key, val):
                stored[(table, key)] = val
                return val

        module = types.ModuleType("tinygrad.engine.search")
        module.diskcache_get, module.diskcache_put = Search.diskcache_get, Search.diskcache_put
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        packages["tinygrad.engine"].search = module
        with unittest.mock.patch.dict(sys.modules, packages):
            counts = bench_common.instrument_tinygrad_beam()
            self.assertEqual(module.diskcache_get("beam_search", "k1"), "winner")
            self.assertIsNone(module.diskcache_get("beam_search", "k2"))
            module.diskcache_put("beam_search", "k2", "found")
            module.diskcache_get("compile", "unrelated")
            module.diskcache_put("compile", "unrelated", "x")

        self.assertEqual(counts, {"call": 0, "hit": 1, "put": 1})
        self.assertIn(("get", "compile", "unrelated"), calls)

    def test_an_uninstrumentable_tinygrad_is_not_an_error(self):
        module = types.ModuleType("tinygrad.engine.search")  # no diskcache helpers
        packages = {
            "tinygrad": types.ModuleType("tinygrad"),
            "tinygrad.engine": types.ModuleType("tinygrad.engine"),
            "tinygrad.engine.search": module,
        }
        with unittest.mock.patch.dict(sys.modules, packages):
            self.assertIsNone(bench_common.instrument_tinygrad_beam())


class FixtureDigestTest(unittest.TestCase):
    """gh-ocannl-645: what a report's numbers were measured on is recorded, not assumed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, name, content=b"weights"):
        path = self.dir / name
        path.write_bytes(content)
        return path

    def check(self, *args, **kwargs):
        """check_fixture_digests with its per-fixture log kept out of the test output."""
        with contextlib.redirect_stdout(io.StringIO()):
            return orchestrate.check_fixture_digests(*args, **kwargs)

    def test_recording_round_trips(self):
        fx = self.fixture("mlp_small.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE

        changes = fixture_digest.record(digests, [fx])

        self.assertEqual([(name, was) for name, was, _ in changes], [("mlp_small.safetensors", None)])
        entries = fixture_digest.read_digests(digests)
        self.assertEqual(entries[fx.name], (fixture_digest.sha256_file(fx), fx.stat().st_size))
        self.assertEqual(fixture_digest.status(fx, entries)[0], "MATCH")

    def test_regenerating_different_bytes_is_announced_as_a_change(self):
        fx = self.fixture("gpt2_mini.safetensors", b"generated at spec revision A")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx])
        was = fixture_digest.read_digests(digests)[fx.name]

        fx.write_bytes(b"generated at spec revision B")
        changes = fixture_digest.record(digests, [fx])

        self.assertEqual(len(changes), 1)
        name, previous, now = changes[0]
        self.assertEqual((name, previous), (fx.name, was))
        self.assertEqual(now, fixture_digest.read_digests(digests)[fx.name])

    def test_regenerating_one_fixture_keeps_the_others_recorded(self):
        # gen_fixtures.py takes a spec list; regenerating one workload must not drop the
        # identities of the fixtures already on disk, or the pin silently narrows to one.
        a, b = self.fixture("a.safetensors", b"aaa"), self.fixture("b.safetensors", b"bbb")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [a, b])

        fixture_digest.record(digests, [a])

        self.assertEqual(set(fixture_digest.read_digests(digests)), {a.name, b.name})

    def test_a_differently_generated_fixture_does_not_match(self):
        # The hazard: the difference is applied uniformly to every cell, so the cross-cell parity
        # gate certifies it. Only the digest contradicts the report's workload name.
        fx = self.fixture("lenet.safetensors", b"the bytes the report was measured on")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx])
        entries = fixture_digest.read_digests(digests)

        fx.write_bytes(b"the bytes someone regenerated later")

        self.assertEqual(fixture_digest.status(fx, entries)[0], "MISMATCH")

    def test_an_unrecorded_fixture_is_not_silently_accepted(self):
        fx = self.fixture("mystery.safetensors")

        self.assertEqual(fixture_digest.status(fx, {})[0], "UNRECORDED")

    def test_a_malformed_line_is_an_error_not_a_dropped_pin(self):
        digests = self.dir / fixture_digest.DIGEST_FILE
        digests.write_text(fixture_digest.HEADER + "deadbeef  lenet.safetensors\n")

        with self.assertRaises(ValueError):
            fixture_digest.read_digests(digests)

    def test_the_checked_in_file_parses_and_names_live_workloads(self):
        # A digest for a workload spec that no longer exists is stale: it pins bytes no current
        # run can produce, and reads as coverage.
        entries = fixture_digest.read_digests(HERE / "fixtures" / fixture_digest.DIGEST_FILE)
        specs = {p.stem for p in (HERE / "workloads").glob("*.json")}
        self.assertTrue(specs, "no workload specs found next to the test")
        for name in entries:
            self.assertTrue(name.endswith(".safetensors"), name)
            self.assertIn(name[: -len(".safetensors")], specs, name)

    def test_the_sweep_refuses_bytes_nothing_records(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.write_digests(digests, {})

        with self.assertRaises(SystemExit) as refused:
            self.check([fx], digests_path=digests)

        self.assertIn("--no-fixture-digest-check", str(refused.exception))

    def test_the_sweep_runs_and_stamps_a_recorded_fixture(self):
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx])

        shas = self.check([fx], digests_path=digests)

        self.assertEqual(shas, {fx: fixture_digest.sha256_file(fx)})

    def test_the_opt_out_measures_them_and_still_reports_the_digest(self):
        # A deliberate regeneration is a legitimate reason to run unpinned; it must not also cost
        # the run its record of what it ran on.
        fx = self.fixture("lenet.safetensors")
        digests = self.dir / fixture_digest.DIGEST_FILE
        fixture_digest.record(digests, [fx])
        fx.write_bytes(b"regenerated")

        shas = self.check([fx], digests_path=digests, allow_unpinned=True)

        self.assertEqual(shas, {fx: fixture_digest.sha256_file(fx)})

    def test_generated_fixtures_are_named_after_their_spec(self):
        # What the recorded-name check above relies on: gen_fixtures.py writes
        # fixtures/<spec name>.safetensors, and every spec's `name` is its file stem.
        for spec in (HERE / "workloads").glob("*.json"):
            self.assertEqual(json.loads(spec.read_text())["name"], spec.stem, spec.name)


if __name__ == "__main__":
    unittest.main()
