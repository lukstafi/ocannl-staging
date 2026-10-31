(* gh-ocannl-1624: the shape-inference state [Ir.Indexing]'s records hold -- a capture reference's
   solution, a static symbol's role -- is written only with the process's one
   [Ir.Indexing.shape_writer]. This executable links no tensor frontend, so no [Shape] claimed the
   writer: the first claim here succeeds (the positive control), the guards behind it are exercised
   directly, and a second claim is refused (the negative control; capture_per_call holds the same
   refusal in a process where [Shape] holds the writer). *)
open Base
open Verdict.Claims
module Idx = Ir.Indexing

let refused f = match f () with () -> false | exception Utils.User_error _ -> true

let with_writer w =
  p "a second claim of the shape writer is refused"
    (match Idx.claim_shape_writer () with
    | (_ : Idx.shape_writer) -> false
    | exception Utils.User_error msg ->
        String.is_substring msg ~substring:"the shape writer is already claimed");
  let seq = Idx.make_static_symbol ~static_range:6 (Idx.get_symbol ()) in
  let r = Idx.variable_ref "r" in
  Idx.set_solved_sym w r seq;
  Idx.set_solved_sym w r seq;
  p "Indexing guard: re-setting the held symbol is a no-op" (Option.is_some r.solved_sym);
  p "Indexing guard: a size, even the symbol's maximum, is refused on a symbol"
    (refused (fun () -> Idx.set_solved_dim w r 6));
  p "Indexing guard: the refused size leaves the symbol alone"
    (Option.is_none r.solved_dim && Option.equal Idx.equal_static_symbol r.solved_sym (Some seq));
  let q = Idx.variable_ref "q" in
  Idx.set_solved_dim w q 6;
  Idx.set_solved_dim w q 6;
  p "Indexing guard: re-setting the held size is a no-op"
    (Option.equal Int.equal q.solved_dim (Some 6));
  p "Indexing guard: a symbol is refused on a size" (refused (fun () -> Idx.set_solved_sym w q seq));
  p "Indexing guard: the refused symbol leaves the size alone"
    (Option.is_none q.solved_sym && Option.equal Int.equal q.solved_dim (Some 6));
  (* Roles: committed once, exclusive of each other. *)
  let ext = Idx.make_static_symbol ~static_range:4 (Idx.get_symbol ()) in
  p "a fresh static symbol holds no role" (not (ext.used_as_extent || ext.used_as_slice));
  Idx.commit_role w ext `Extent;
  Idx.commit_role w ext `Extent;
  p "committing the extent role sets it, and re-committing it is a no-op"
    (ext.used_as_extent && not ext.used_as_slice);
  p "the slice role is refused on an extent symbol, leaving it an extent"
    (refused (fun () -> Idx.commit_role w ext `Slice) && ext.used_as_extent && not ext.used_as_slice);
  let born = Idx.make_static_symbol ~static_range:4 ~used_as:`Slice (Idx.get_symbol ()) in
  p "a role given at birth holds, and excludes the other"
    (born.used_as_slice && (not born.used_as_extent)
    && refused (fun () -> Idx.commit_role w born `Extent))

let () =
  let claimed =
    match Idx.claim_shape_writer () with w -> Ok w | exception Utils.User_error msg -> Error msg
  in
  p "with no Shape linked, the first claim of the shape writer succeeds" (Result.is_ok claimed);
  (* Every later leg needs the writer; a failed claim is already the failed verdict above. *)
  Result.iter claimed ~f:with_writer
