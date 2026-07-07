(* Tutorial: GPT-2 124M inference with pretrained weights, end to end in OCANNL.

   Downloads (and caches under ~/.cache/ocannl/) the HuggingFace GPT-2 tokenizer and
   [model.safetensors] (~500MB) on first run, builds the model at a fixed window length, compiles
   one forward routine, and greedy-decodes a continuation of the prompt. The window is recomputed
   each step (no KV cache): with a causal mask, logits at position [len - 1] are unaffected by the
   padding that follows, so one compiled routine serves the whole generation.

   Run: dune exec test/gpt2/gpt2_generate.exe -- "Your prompt here" [num_tokens] *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules

let model_id = "openai-community/gpt2"
let eos = 50256

let fetch_weights () =
  let url = [%string "https://huggingface.co/%{model_id}/resolve/main/model.safetensors"] in
  let dest = Dataprep.Dataset_utils.get_cache_dir "models" ^ "gpt2/model.safetensors" in
  Dataprep.Dataset_utils.ensure_file url dest;
  Safetensors.read dest

let () =
  let argv = Sys.get_argv () in
  let prompt = if Array.length argv > 1 then argv.(1) else "The capital of France is" in
  let num_tokens = if Array.length argv > 2 then Int.of_string argv.(2) else 20 in
  let tokenizer = Dataprep.Bpe.from_pretrained model_id in
  let prompt_ids = Dataprep.Bpe.encode tokenizer prompt in
  let prompt_len = Array.length prompt_ids in
  let seq_len = prompt_len + num_tokens in
  let config = Gpt2_model.gpt2_small in
  assert (prompt_len > 0 && seq_len <= config.n_positions);
  printf "Loading weights...\n%!";
  let st = fetch_weights () in
  printf "Building GPT-2 (%d-token window)...\n%!" seq_len;
  let model = Gpt2_model.gpt2 ~config ~src:(Gpt2_model.Pretrained st) ~seq_len () in
  let ids = Nn_blocks.token_ids_of_array ~max_len:seq_len ~pad_id:eos prompt_ids in
  let logits = model ids in
  printf "Compiling...\n%!";
  let comp = Train.forward logits in
  Train.set_materialized ids.Tensor.value;
  let ctx = Context.auto () in
  let _ctx, routine = Context.compile ctx comp Train.IDX.empty in
  let ctx = Context.context routine in
  let current = Array.append prompt_ids (Array.create ~len:num_tokens eos) in
  printf "%s%!" prompt;
  let rec step len ctx =
    if len < seq_len then begin
      Train.run ctx routine;
      (* Greedy: argmax of the logits row at the last real position. *)
      let vals = Context.get_values ctx logits.Tensor.value in
      let row = (len - 1) * config.vocab_size in
      let next = ref 0 and best = ref Float.neg_infinity in
      for v = 0 to config.vocab_size - 1 do
        let l = vals.(row + v) in
        if Float.(l > !best) then begin
          best := l;
          next := v
        end
      done;
      if !next <> eos then begin
        printf "%s%!" (Dataprep.Bpe.decode tokenizer [| !next |]);
        current.(len) <- !next;
        let ctx = Context.set_values ctx ids.Tensor.value (Array.map current ~f:Float.of_int) in
        step (len + 1) ctx
      end
    end
  in
  step prompt_len ctx;
  printf "\n%!"
