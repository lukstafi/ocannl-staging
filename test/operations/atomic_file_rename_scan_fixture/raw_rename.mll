{
(* Permanent negative control for gh-ocannl-862. This raw publication must be found in the
   generated [raw_rename.ml] through the dependency set derived from this input. *)
let publish source target = Sys.rename source target
}

rule token = parse
| eof { () }
