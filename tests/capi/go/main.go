// Package main is a smoke test proving snapjudge's C ABI is callable from Go
// via cgo. It links the static libsnapjudge.a and runs one typed prediction.
//
// Build (from the repo root):
//
//	go build -o /tmp/sj-go -ldflags "-linkmode external" ./tests/capi/go
//
// The cgo preamble points at the built static lib + headers.
package main

/*
#cgo CFLAGS: -I${SRCDIR}/../../../include -I${SRCDIR}/../../../third_party
#cgo LDFLAGS: -L${SRCDIR}/../../../build -lsnapjudge -framework Accelerate -lpcre2-8 -lcurl -lstdc++
#include <stdlib.h>
#include "snapjudge/capi.h"
*/
import "C"

import (
	"fmt"
	"os"
	"unsafe"
)

func main() {
	ckpt := os.Getenv("SNAPJUDGE_TINY_CKPT")
	if ckpt == "" {
		ckpt = "build/tiny-ckpt"
	}

	e := C.sj_engine_new(nil)
	if e == nil {
		fmt.Fprintln(os.Stderr, "FAIL: engine_new")
		os.Exit(1)
	}
	defer C.sj_engine_free(e)

	cname := C.CString("main")
	cpath := C.CString(ckpt)
	defer C.free(unsafe.Pointer(cname))
	defer C.free(unsafe.Pointer(cpath))
	if C.sj_load(e, cname, cpath) != 0 {
		fmt.Fprintln(os.Stderr, "FAIL: sj_load")
		os.Exit(1)
	}

	state := C.CString(`{"text":"I was charged twice, please refund"}`)
	questions := C.CString(`{"q":{"type":"noul","instructions":"Does the customer want money back?"}}`)
	defer C.free(unsafe.Pointer(state))
	defer C.free(unsafe.Pointer(questions))

	out := C.sj_predict(e, state, questions, nil)
	if out == nil {
		fmt.Fprintln(os.Stderr, "FAIL: sj_predict")
		os.Exit(1)
	}
	defer C.sj_string_free(out)

	fmt.Println("Go cgo OK")
	fmt.Println(C.GoString(out))
}
