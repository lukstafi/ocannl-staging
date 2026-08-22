(* gh-ocannl-685: the action menu's per-unit cap is shared across its categories, not spent in
   category order.

   [Autotune.menu] builds five category lists — tensorizes, splits, swaps, unrolls, vectorizes —
   and used to cap them with [List.take (tensorizes @ splits @ ...) max_actions_per_unit]. The list
   is a concatenation ordered by category and is NOT ranked (contrast the placement surface's
   prefix over [rank_flip_candidates], where top-N is the intended semantics), so the prefix was
   arbitrary: a unit whose tensorizes alone reach the cap — tensorize proposals are up to 6 role
   assignments per tightly-nested innermost serial triple, so the count scales with the number of
   matmul-shaped nests — offered the search no split, swap, unroll or vectorize action at all. Not
   fewer of them: none. And those are exactly the categories a unit needs when its tensorizes turn
   out [Op_illegal] or unprofitable. Silent search-space loss, with a [menu:] log line reporting
   the counts BEFORE the take, so a truncated menu logged the same numbers as an untruncated one.

   [Autotune.share_cap] is the fix and this is its pin, over synthetic categories (integers stand
   in for proposals) so the claims are about the sharing rule rather than about whichever menu a
   particular lowering happens to produce. The negative control is stated as its own claim: the
   plain prefix the cap used to be starves every category after the first on the same input, so a
   revert fails the representation claim rather than quietly passing. *)

open Base
module V = Verdict

let p = V.p
let cap = 48

(* Category sizes: one overflowing category, one comfortably small, one empty, and two in
   between — 100 + 6 + 0 + 10 + 5 = 121 proposals competing for 48 slots. *)
let sizes = [ ("tensorize", 100); ("split", 6); ("swap", 0); ("unroll", 10); ("vectorize", 5) ]

(* Each synthetic proposal remembers its category, so "represented" is checkable. *)
let categories = List.map sizes ~f:(fun (name, n) -> (name, List.init n ~f:(fun i -> (name, i))))
let total = List.sum (module Int) sizes ~f:snd
let non_empty = List.filter_map sizes ~f:(fun (name, n) -> if n > 0 then Some name else None)
let represented kept name = List.exists kept ~f:(fun (c, _) -> String.equal c name)

let () =
  let kept, dropped = Autotune.share_cap ~cap categories in
  p "the overflowing menu is capped to exactly the budget" (List.length kept = cap);
  p "every non-empty category is represented under the cap"
    (List.for_all non_empty ~f:(represented kept));
  p "an empty category is not conjured into the menu" (not (represented kept "swap"));
  (* Remainder spill: with 4 non-empty categories an equal share is 12, so the two categories
     smaller than that keep everything and their unused share goes to the larger ones. *)
  let kept_of name = List.count kept ~f:(fun (c, _) -> String.equal c name) in
  p "a category smaller than its share is kept whole (split: 6 of 6)" (kept_of "split" = 6);
  p "a category smaller than its share is kept whole (vectorize: 5 of 5)"
    (kept_of "vectorize" = 5);
  p "the spilled remainder goes to the categories that still have proposals"
    (kept_of "tensorize" + kept_of "unroll" = cap - 6 - 5);
  p "no category is starved to zero while it has proposals and budget remains"
    (List.for_all non_empty ~f:(fun name -> kept_of name > 0));
  (* Survivors keep their category order and their within-category order, so the fix is invisible
     to anything downstream that fits under the cap. *)
  p "survivors stay in category order and in each category's own order"
    (List.equal
       (fun (a, i) (b, j) -> String.equal a b && i = j)
       kept
       (List.concat_map sizes ~f:(fun (name, _) ->
            List.filter kept ~f:(fun (c, _) -> String.equal c name))));
  (* What the cap dropped is reported, per category, rather than what it was offered. *)
  p "the drop report accounts for every withheld proposal"
    (List.sum (module Int) dropped ~f:snd = total - cap);
  p "the drop report names only categories that actually lost proposals"
    (List.for_all dropped ~f:(fun (name, d) ->
         d > 0 && d = List.Assoc.find_exn sizes name ~equal:String.equal - kept_of name));
  (* An under-full menu is returned untouched, with nothing reported dropped. *)
  let small = List.map sizes ~f:(fun (name, n) -> (name, List.init (min n 3) ~f:(fun i -> (name, i)))) in
  let small_kept, small_dropped = Autotune.share_cap ~cap small in
  p "a menu that fits under the cap is unchanged"
    (List.equal
       (fun (a, i) (b, j) -> String.equal a b && i = j)
       small_kept
       (List.concat_map small ~f:snd));
  p "a menu that fits under the cap reports no drops" (List.is_empty small_dropped);
  (* The negative control, stated as a claim: on this very input the plain prefix the cap used to
     be represents the first category and nothing else. *)
  let prefix = List.take (List.concat_map categories ~f:snd) cap in
  p "the plain prefix this replaces would have starved every category after the first"
    (List.for_all non_empty ~f:(fun name ->
         String.equal name "tensorize" || not (represented prefix name)));
  Stdio.printf "\nDone.\n%!"
