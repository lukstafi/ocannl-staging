#include <caml/mlvalues.h>
#include <stdio.h>

CAMLprim value ocannl_flush_c_streams(value unit) {
  (void)unit;
  fflush(NULL);
  return Val_unit;
}
