/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) id_spring_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s2[15] = {2, 4, 0, 2, 4, 6, 8, 0, 1, 0, 1, 0, 1, 0, 1};

/* id_spring_controller:(i0[4])->(o0[2]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a4, a5, a6, a7, a8, a9;
  a0=4.1193000000000002e-04;
  a1=-1.6000000000000000e-01;
  a2=arg[0]? arg[0][0] : 0;
  a3=sin(a2);
  a4=(a1*a3);
  a5=arg[0]? arg[0][1] : 0;
  a6=cos(a5);
  a7=(a6*a3);
  a8=sin(a5);
  a9=cos(a2);
  a10=(a8*a9);
  a7=(a7+a10);
  a7=(a1*a7);
  a4=(a4+a7);
  a7=casadi_sq(a4);
  a10=cos(a2);
  a11=sin(a5);
  a12=(a10*a11);
  a13=sin(a2);
  a14=cos(a5);
  a15=(a13*a14);
  a12=(a12+a15);
  a12=(a1*a12);
  a15=casadi_sq(a12);
  a7=(a7+a15);
  a15=cos(a2);
  a16=(a1*a15);
  a17=cos(a5);
  a18=(a17*a15);
  a19=sin(a5);
  a20=sin(a2);
  a21=(a19*a20);
  a18=(a18-a21);
  a18=(a1*a18);
  a16=(a16+a18);
  a18=casadi_sq(a16);
  a21=cos(a2);
  a22=cos(a5);
  a23=(a21*a22);
  a24=sin(a2);
  a25=sin(a5);
  a26=(a24*a25);
  a23=(a23-a26);
  a23=(a1*a23);
  a26=casadi_sq(a23);
  a18=(a18+a26);
  a26=(a18*a7);
  a27=(a16*a4);
  a28=(a23*a12);
  a27=(a27+a28);
  a28=(a4*a16);
  a29=(a12*a23);
  a28=(a28+a29);
  a29=(a27*a28);
  a26=(a26-a29);
  a7=(a7/a26);
  a29=(a7*a16);
  a27=(a27/a26);
  a30=(a27*a4);
  a29=(a29-a30);
  a30=5.;
  a31=-15.;
  a32=(a1*a24);
  a33=(a21*a19);
  a34=(a24*a17);
  a33=(a33+a34);
  a33=(a1*a33);
  a32=(a32+a33);
  a33=1.9533952097487400e-02;
  a32=(a32-a33);
  a32=(a31*a32);
  a33=2.9999999999999999e-01;
  a34=arg[0]? arg[0][2] : 0;
  a35=(a16*a34);
  a36=arg[0]? arg[0][3] : 0;
  a37=(a23*a36);
  a35=(a35+a37);
  a35=(a33*a35);
  a32=(a32-a35);
  a32=(a30*a32);
  a35=sin(a2);
  a35=(a35*a34);
  a37=(a1*a35);
  a38=sin(a5);
  a38=(a38*a36);
  a15=(a15*a38);
  a17=(a17*a35);
  a15=(a15+a17);
  a17=cos(a5);
  a17=(a17*a36);
  a20=(a20*a17);
  a17=cos(a2);
  a17=(a17*a34);
  a19=(a19*a17);
  a20=(a20+a19);
  a15=(a15+a20);
  a15=(a1*a15);
  a37=(a37+a15);
  a37=(a37*a34);
  a15=sin(a2);
  a15=(a15*a34);
  a22=(a22*a15);
  a15=sin(a5);
  a15=(a15*a36);
  a21=(a21*a15);
  a22=(a22+a21);
  a21=cos(a2);
  a21=(a21*a34);
  a25=(a25*a21);
  a21=cos(a5);
  a21=(a21*a36);
  a24=(a24*a21);
  a25=(a25+a24);
  a22=(a22+a25);
  a22=(a1*a22);
  a22=(a22*a36);
  a37=(a37+a22);
  a32=(a32+a37);
  a29=(a29*a32);
  a28=(a28/a26);
  a16=(a28*a16);
  a18=(a18/a26);
  a26=(a18*a4);
  a16=(a16-a26);
  a26=(a1*a10);
  a37=(a10*a6);
  a22=(a13*a8);
  a37=(a37-a22);
  a37=(a1*a37);
  a26=(a26+a37);
  a37=-1.2924818151883638e-01;
  a26=(a26-a37);
  a31=(a31*a26);
  a4=(a4*a34);
  a26=(a12*a36);
  a4=(a4+a26);
  a33=(a33*a4);
  a31=(a31+a33);
  a30=(a30*a31);
  a31=cos(a2);
  a31=(a31*a34);
  a33=(a1*a31);
  a6=(a6*a31);
  a31=sin(a5);
  a31=(a31*a36);
  a3=(a3*a31);
  a6=(a6-a3);
  a3=cos(a5);
  a3=(a3*a36);
  a9=(a9*a3);
  a3=sin(a2);
  a3=(a3*a34);
  a8=(a8*a3);
  a9=(a9-a8);
  a6=(a6+a9);
  a6=(a1*a6);
  a33=(a33+a6);
  a33=(a33*a34);
  a6=cos(a5);
  a6=(a6*a36);
  a10=(a10*a6);
  a6=sin(a2);
  a6=(a6*a34);
  a11=(a11*a6);
  a10=(a10-a11);
  a11=cos(a2);
  a11=(a11*a34);
  a14=(a14*a11);
  a11=sin(a5);
  a11=(a11*a36);
  a13=(a13*a11);
  a14=(a14-a13);
  a10=(a10+a14);
  a10=(a1*a10);
  a10=(a10*a36);
  a33=(a33+a10);
  a30=(a30+a33);
  a16=(a16*a30);
  a29=(a29+a16);
  a0=(a0*a29);
  a16=-7.8706999999999999e-02;
  a33=1.4853844999999999e-01;
  a10=(a16*a29);
  a14=9.8100000000000005e+00;
  a13=sin(a2);
  a13=(a14*a13);
  a10=(a10-a13);
  a10=(a33*a10);
  a16=(a16*a10);
  a10=-1.3770000000000000e-05;
  a2=cos(a2);
  a14=(a14*a2);
  a2=(a10*a29);
  a2=(a14-a2);
  a33=(a33*a2);
  a10=(a10*a33);
  a16=(a16-a10);
  a0=(a0+a16);
  a16=1.4913921739033597e-04;
  a7=(a7*a23);
  a27=(a27*a12);
  a7=(a7-a27);
  a7=(a7*a32);
  a28=(a28*a23);
  a18=(a18*a12);
  a28=(a28-a18);
  a28=(a28*a30);
  a7=(a7+a28);
  a7=(a7+a29);
  a16=(a16*a7);
  a28=-1.0224903071020168e-01;
  a30=3.7636070000000001e-02;
  a18=cos(a5);
  a29=(a1*a29);
  a29=(a29-a13);
  a13=(a18*a29);
  a5=sin(a5);
  a12=(a5*a14);
  a13=(a13-a12);
  a12=(a1*a34);
  a23=(a5*a12);
  a32=(a36*a23);
  a13=(a13-a32);
  a7=(a28*a7);
  a13=(a13+a7);
  a13=(a30*a13);
  a7=(a28*a13);
  a16=(a16+a7);
  a12=(a18*a12);
  a34=(a36+a34);
  a28=(a28*a34);
  a28=(a12+a28);
  a28=(a30*a28);
  a7=(a23*a28);
  a23=(a30*a23);
  a32=(a12*a23);
  a7=(a7-a32);
  a16=(a16+a7);
  a23=(a34*a23);
  a13=(a13+a23);
  a13=(a18*a13);
  a36=(a36*a12);
  a29=(a5*a29);
  a18=(a18*a14);
  a29=(a29+a18);
  a36=(a36+a29);
  a30=(a30*a36);
  a34=(a34*a28);
  a30=(a30-a34);
  a5=(a5*a30);
  a13=(a13+a5);
  a1=(a1*a13);
  a1=(a16+a1);
  a0=(a0+a1);
  if (res[0]!=0) res[0][0]=a0;
  if (res[0]!=0) res[0][1]=a16;
  return 0;
}

CASADI_SYMBOL_EXPORT int id_spring_controller(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int id_spring_controller_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int id_spring_controller_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void id_spring_controller_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int id_spring_controller_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void id_spring_controller_release(int mem) {
}

CASADI_SYMBOL_EXPORT void id_spring_controller_incref(void) {
}

CASADI_SYMBOL_EXPORT void id_spring_controller_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int id_spring_controller_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int id_spring_controller_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real id_spring_controller_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* id_spring_controller_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* id_spring_controller_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* id_spring_controller_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* id_spring_controller_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int id_spring_controller_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int id_spring_controller_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}

/* delta_controller_delta_x:(i0[4])->(o0[2x4]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a100, a101, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a87, a88, a89, a9, a90, a91, a92, a93, a94, a95, a96, a97, a98, a99;
  a0=4.1193000000000002e-04;
  a1=5.;
  a2=-15.;
  a3=-1.6000000000000000e-01;
  a4=arg[0]? arg[0][0] : 0;
  a5=sin(a4);
  a6=(a3*a5);
  a7=cos(a4);
  a8=arg[0]? arg[0][1] : 0;
  a9=sin(a8);
  a10=(a7*a9);
  a11=cos(a8);
  a12=(a5*a11);
  a10=(a10+a12);
  a10=(a3*a10);
  a6=(a6+a10);
  a10=1.9533952097487400e-02;
  a6=(a6-a10);
  a6=(a2*a6);
  a10=2.9999999999999999e-01;
  a12=cos(a4);
  a13=(a3*a12);
  a14=(a11*a12);
  a15=sin(a4);
  a16=(a9*a15);
  a14=(a14-a16);
  a14=(a3*a14);
  a13=(a13+a14);
  a14=arg[0]? arg[0][2] : 0;
  a16=(a13*a14);
  a17=cos(a8);
  a18=(a7*a17);
  a19=sin(a8);
  a20=(a5*a19);
  a18=(a18-a20);
  a18=(a3*a18);
  a20=arg[0]? arg[0][3] : 0;
  a21=(a18*a20);
  a16=(a16+a21);
  a16=(a10*a16);
  a6=(a6-a16);
  a6=(a1*a6);
  a16=sin(a4);
  a21=(a16*a14);
  a22=(a3*a21);
  a23=sin(a8);
  a24=(a23*a20);
  a25=(a12*a24);
  a26=(a11*a21);
  a25=(a25+a26);
  a26=cos(a8);
  a27=(a26*a20);
  a28=(a15*a27);
  a29=cos(a4);
  a30=(a29*a14);
  a31=(a9*a30);
  a28=(a28+a31);
  a25=(a25+a28);
  a25=(a3*a25);
  a22=(a22+a25);
  a25=(a22*a14);
  a28=sin(a4);
  a31=(a28*a14);
  a32=(a17*a31);
  a33=sin(a8);
  a34=(a33*a20);
  a35=(a7*a34);
  a32=(a32+a35);
  a35=cos(a4);
  a36=(a35*a14);
  a37=(a19*a36);
  a38=cos(a8);
  a39=(a38*a20);
  a40=(a5*a39);
  a37=(a37+a40);
  a32=(a32+a37);
  a32=(a3*a32);
  a37=(a32*a20);
  a25=(a25+a37);
  a6=(a6+a25);
  a25=sin(a4);
  a37=(a3*a25);
  a40=cos(a8);
  a41=(a40*a25);
  a42=sin(a8);
  a43=cos(a4);
  a44=(a42*a43);
  a41=(a41+a44);
  a41=(a3*a41);
  a37=(a37+a41);
  a41=(a37+a37);
  a44=cos(a4);
  a45=(a3*a44);
  a46=(a40*a44);
  a47=sin(a4);
  a48=(a42*a47);
  a46=(a46-a48);
  a46=(a3*a46);
  a45=(a45+a46);
  a46=(a41*a45);
  a48=cos(a4);
  a49=sin(a8);
  a50=(a48*a49);
  a51=sin(a4);
  a52=cos(a8);
  a53=(a51*a52);
  a50=(a50+a53);
  a50=(a3*a50);
  a53=(a50+a50);
  a54=cos(a4);
  a55=(a52*a54);
  a56=sin(a4);
  a57=(a49*a56);
  a55=(a55-a57);
  a55=(a3*a55);
  a57=(a53*a55);
  a46=(a46+a57);
  a57=casadi_sq(a13);
  a58=casadi_sq(a18);
  a57=(a57+a58);
  a58=casadi_sq(a37);
  a59=casadi_sq(a50);
  a58=(a58+a59);
  a59=(a57*a58);
  a60=(a13*a37);
  a61=(a18*a50);
  a60=(a60+a61);
  a61=(a37*a13);
  a62=(a50*a18);
  a61=(a61+a62);
  a62=(a60*a61);
  a59=(a59-a62);
  a62=(a46/a59);
  a63=(a58/a59);
  a64=(a63/a59);
  a46=(a57*a46);
  a65=(a13+a13);
  a66=sin(a4);
  a67=(a3*a66);
  a68=(a11*a66);
  a69=cos(a4);
  a70=(a9*a69);
  a68=(a68+a70);
  a68=(a3*a68);
  a67=(a67+a68);
  a68=(a65*a67);
  a70=(a18+a18);
  a71=sin(a4);
  a72=(a17*a71);
  a73=cos(a4);
  a74=(a19*a73);
  a72=(a72+a74);
  a72=(a3*a72);
  a74=(a70*a72);
  a68=(a68+a74);
  a74=(a58*a68);
  a46=(a46-a74);
  a74=(a13*a45);
  a75=(a37*a67);
  a74=(a74-a75);
  a75=(a18*a55);
  a76=(a50*a72);
  a75=(a75-a76);
  a74=(a74+a75);
  a75=(a61*a74);
  a76=(a13*a45);
  a77=(a37*a67);
  a76=(a76-a77);
  a77=(a18*a55);
  a78=(a50*a72);
  a77=(a77-a78);
  a76=(a76+a77);
  a77=(a60*a76);
  a75=(a75+a77);
  a46=(a46-a75);
  a75=(a64*a46);
  a62=(a62-a75);
  a75=(a13*a62);
  a77=(a63*a67);
  a75=(a75-a77);
  a74=(a74/a59);
  a77=(a60/a59);
  a78=(a77/a59);
  a79=(a78*a46);
  a74=(a74-a79);
  a79=(a37*a74);
  a80=(a77*a45);
  a79=(a79+a80);
  a75=(a75-a79);
  a75=(a6*a75);
  a79=(a63*a13);
  a80=(a77*a37);
  a79=(a79-a80);
  a80=(a3*a73);
  a81=(a11*a73);
  a82=(a9*a71);
  a81=(a81-a82);
  a81=(a3*a81);
  a80=(a80+a81);
  a80=(a2*a80);
  a81=(a14*a67);
  a82=(a20*a72);
  a81=(a81+a82);
  a81=(a10*a81);
  a80=(a80+a81);
  a80=(a1*a80);
  a81=cos(a4);
  a81=(a14*a81);
  a82=(a3*a81);
  a81=(a11*a81);
  a24=(a24*a66);
  a81=(a81-a24);
  a27=(a27*a69);
  a69=sin(a4);
  a69=(a14*a69);
  a69=(a9*a69);
  a27=(a27-a69);
  a81=(a81+a27);
  a81=(a3*a81);
  a82=(a82+a81);
  a82=(a14*a82);
  a81=cos(a4);
  a81=(a14*a81);
  a81=(a17*a81);
  a34=(a34*a71);
  a81=(a81-a34);
  a39=(a39*a73);
  a73=sin(a4);
  a73=(a14*a73);
  a73=(a19*a73);
  a39=(a39-a73);
  a81=(a81+a39);
  a81=(a3*a81);
  a81=(a20*a81);
  a82=(a82+a81);
  a80=(a80+a82);
  a82=(a79*a80);
  a75=(a75+a82);
  a82=(a3*a48);
  a81=(a48*a40);
  a39=(a51*a42);
  a81=(a81-a39);
  a81=(a3*a81);
  a82=(a82+a81);
  a81=-1.2924818151883638e-01;
  a82=(a82-a81);
  a82=(a2*a82);
  a81=(a37*a14);
  a39=(a50*a20);
  a81=(a81+a39);
  a81=(a10*a81);
  a82=(a82+a81);
  a82=(a1*a82);
  a81=cos(a4);
  a39=(a81*a14);
  a73=(a3*a39);
  a34=(a40*a39);
  a71=sin(a8);
  a27=(a71*a20);
  a69=(a25*a27);
  a34=(a34-a69);
  a69=cos(a8);
  a24=(a69*a20);
  a66=(a43*a24);
  a83=sin(a4);
  a84=(a83*a14);
  a85=(a42*a84);
  a66=(a66-a85);
  a34=(a34+a66);
  a34=(a3*a34);
  a73=(a73+a34);
  a34=(a73*a14);
  a66=cos(a8);
  a85=(a66*a20);
  a86=(a48*a85);
  a87=sin(a4);
  a88=(a87*a14);
  a89=(a49*a88);
  a86=(a86-a89);
  a89=cos(a4);
  a90=(a89*a14);
  a91=(a52*a90);
  a92=sin(a8);
  a93=(a92*a20);
  a94=(a51*a93);
  a91=(a91-a94);
  a86=(a86+a91);
  a86=(a3*a86);
  a91=(a86*a20);
  a34=(a34+a91);
  a82=(a82+a34);
  a76=(a76/a59);
  a34=(a61/a59);
  a91=(a34/a59);
  a94=(a91*a46);
  a76=(a76-a94);
  a94=(a13*a76);
  a67=(a34*a67);
  a94=(a94-a67);
  a67=(a57/a59);
  a95=(a67*a45);
  a68=(a68/a59);
  a96=(a67/a59);
  a46=(a96*a46);
  a68=(a68+a46);
  a46=(a37*a68);
  a95=(a95-a46);
  a94=(a94-a95);
  a94=(a82*a94);
  a95=(a34*a13);
  a46=(a67*a37);
  a95=(a95-a46);
  a45=(a14*a45);
  a46=(a20*a55);
  a45=(a45+a46);
  a45=(a10*a45);
  a46=(a3*a56);
  a97=(a40*a56);
  a98=(a42*a54);
  a97=(a97+a98);
  a97=(a3*a97);
  a46=(a46+a97);
  a46=(a2*a46);
  a45=(a45-a46);
  a45=(a1*a45);
  a46=sin(a4);
  a46=(a14*a46);
  a97=(a3*a46);
  a46=(a40*a46);
  a27=(a27*a44);
  a46=(a46+a27);
  a24=(a24*a47);
  a47=cos(a4);
  a47=(a14*a47);
  a47=(a42*a47);
  a24=(a24+a47);
  a46=(a46+a24);
  a46=(a3*a46);
  a97=(a97+a46);
  a97=(a14*a97);
  a85=(a85*a56);
  a56=cos(a4);
  a56=(a14*a56);
  a56=(a49*a56);
  a85=(a85+a56);
  a56=sin(a4);
  a56=(a14*a56);
  a56=(a52*a56);
  a93=(a93*a54);
  a56=(a56+a93);
  a85=(a85+a56);
  a85=(a3*a85);
  a85=(a20*a85);
  a97=(a97+a85);
  a45=(a45-a97);
  a97=(a95*a45);
  a94=(a94+a97);
  a75=(a75+a94);
  a94=(a0*a75);
  a97=-7.8706999999999999e-02;
  a85=1.4853844999999999e-01;
  a56=(a97*a75);
  a93=9.8100000000000005e+00;
  a54=cos(a4);
  a54=(a93*a54);
  a56=(a56-a54);
  a56=(a85*a56);
  a56=(a97*a56);
  a46=-1.3770000000000000e-05;
  a24=sin(a4);
  a24=(a93*a24);
  a47=(a46*a75);
  a47=(a24+a47);
  a47=(a85*a47);
  a47=(a46*a47);
  a56=(a56+a47);
  a94=(a94+a56);
  a56=1.4913921739033597e-04;
  a62=(a18*a62);
  a47=(a63*a72);
  a62=(a62-a47);
  a74=(a50*a74);
  a47=(a77*a55);
  a74=(a74+a47);
  a62=(a62-a74);
  a62=(a6*a62);
  a74=(a63*a18);
  a47=(a77*a50);
  a74=(a74-a47);
  a80=(a74*a80);
  a62=(a62+a80);
  a76=(a18*a76);
  a72=(a34*a72);
  a76=(a76-a72);
  a55=(a67*a55);
  a68=(a50*a68);
  a55=(a55-a68);
  a76=(a76-a55);
  a76=(a82*a76);
  a55=(a34*a18);
  a68=(a67*a50);
  a55=(a55-a68);
  a45=(a55*a45);
  a76=(a76+a45);
  a62=(a62+a76);
  a62=(a62+a75);
  a76=(a56*a62);
  a45=-1.0224903071020168e-01;
  a68=3.7636070000000001e-02;
  a72=cos(a8);
  a75=(a3*a75);
  a75=(a75-a54);
  a54=(a72*a75);
  a80=sin(a8);
  a47=(a80*a24);
  a54=(a54+a47);
  a62=(a45*a62);
  a54=(a54+a62);
  a54=(a68*a54);
  a62=(a45*a54);
  a76=(a76+a62);
  a54=(a72*a54);
  a75=(a80*a75);
  a24=(a72*a24);
  a75=(a75-a24);
  a75=(a68*a75);
  a75=(a80*a75);
  a54=(a54+a75);
  a54=(a3*a54);
  a54=(a76+a54);
  a94=(a94+a54);
  if (res[0]!=0) res[0][0]=a94;
  if (res[0]!=0) res[0][1]=a76;
  a76=cos(a8);
  a94=(a43*a76);
  a54=sin(a8);
  a75=(a25*a54);
  a94=(a94-a75);
  a94=(a3*a94);
  a41=(a41*a94);
  a75=cos(a8);
  a24=(a48*a75);
  a62=sin(a8);
  a47=(a51*a62);
  a24=(a24-a47);
  a24=(a3*a24);
  a53=(a53*a24);
  a41=(a41+a53);
  a53=(a41/a59);
  a57=(a57*a41);
  a41=sin(a8);
  a47=(a12*a41);
  a27=cos(a8);
  a44=(a15*a27);
  a47=(a47+a44);
  a47=(a3*a47);
  a65=(a65*a47);
  a44=sin(a8);
  a98=(a7*a44);
  a99=cos(a8);
  a100=(a5*a99);
  a98=(a98+a100);
  a98=(a3*a98);
  a70=(a70*a98);
  a65=(a65+a70);
  a58=(a58*a65);
  a57=(a57-a58);
  a58=(a13*a94);
  a70=(a37*a47);
  a58=(a58-a70);
  a70=(a18*a24);
  a100=(a50*a98);
  a70=(a70-a100);
  a58=(a58+a70);
  a61=(a61*a58);
  a70=(a13*a94);
  a100=(a37*a47);
  a70=(a70-a100);
  a100=(a18*a24);
  a101=(a50*a98);
  a100=(a100-a101);
  a70=(a70+a100);
  a60=(a60*a70);
  a61=(a61+a60);
  a57=(a57-a61);
  a64=(a64*a57);
  a53=(a53-a64);
  a64=(a13*a53);
  a61=(a63*a47);
  a64=(a64-a61);
  a58=(a58/a59);
  a78=(a78*a57);
  a58=(a58-a78);
  a78=(a37*a58);
  a61=(a77*a94);
  a78=(a78+a61);
  a64=(a64-a78);
  a64=(a6*a64);
  a78=(a7*a27);
  a61=(a5*a41);
  a78=(a78-a61);
  a78=(a3*a78);
  a78=(a2*a78);
  a61=(a14*a47);
  a60=(a20*a98);
  a61=(a61+a60);
  a61=(a10*a61);
  a78=(a78+a61);
  a78=(a1*a78);
  a61=cos(a8);
  a61=(a20*a61);
  a61=(a12*a61);
  a21=(a21*a41);
  a61=(a61-a21);
  a30=(a30*a27);
  a27=sin(a8);
  a27=(a20*a27);
  a27=(a15*a27);
  a30=(a30-a27);
  a61=(a61+a30);
  a61=(a3*a61);
  a61=(a14*a61);
  a30=cos(a8);
  a30=(a20*a30);
  a30=(a7*a30);
  a31=(a31*a44);
  a30=(a30-a31);
  a36=(a36*a99);
  a99=sin(a8);
  a99=(a20*a99);
  a99=(a5*a99);
  a36=(a36-a99);
  a30=(a30+a36);
  a30=(a3*a30);
  a30=(a20*a30);
  a61=(a61+a30);
  a78=(a78+a61);
  a61=(a79*a78);
  a64=(a64+a61);
  a70=(a70/a59);
  a91=(a91*a57);
  a70=(a70-a91);
  a91=(a13*a70);
  a47=(a34*a47);
  a91=(a91-a47);
  a47=(a67*a94);
  a65=(a65/a59);
  a96=(a96*a57);
  a65=(a65+a96);
  a96=(a37*a65);
  a47=(a47-a96);
  a91=(a91-a47);
  a91=(a82*a91);
  a94=(a14*a94);
  a47=(a20*a24);
  a94=(a94+a47);
  a94=(a10*a94);
  a47=(a48*a54);
  a96=(a51*a76);
  a47=(a47+a96);
  a47=(a3*a47);
  a2=(a2*a47);
  a94=(a94-a2);
  a94=(a1*a94);
  a39=(a39*a54);
  a54=cos(a8);
  a54=(a20*a54);
  a54=(a25*a54);
  a39=(a39+a54);
  a54=sin(a8);
  a54=(a20*a54);
  a54=(a43*a54);
  a84=(a84*a76);
  a54=(a54+a84);
  a39=(a39+a54);
  a39=(a3*a39);
  a39=(a14*a39);
  a54=sin(a8);
  a54=(a20*a54);
  a54=(a48*a54);
  a88=(a88*a75);
  a54=(a54+a88);
  a90=(a90*a62);
  a62=cos(a8);
  a62=(a20*a62);
  a62=(a51*a62);
  a90=(a90+a62);
  a54=(a54+a90);
  a54=(a3*a54);
  a54=(a20*a54);
  a39=(a39+a54);
  a94=(a94-a39);
  a39=(a95*a94);
  a91=(a91+a39);
  a64=(a64+a91);
  a91=(a0*a64);
  a39=(a97*a64);
  a39=(a85*a39);
  a39=(a97*a39);
  a54=(a46*a64);
  a54=(a85*a54);
  a54=(a46*a54);
  a39=(a39+a54);
  a91=(a91+a39);
  a53=(a18*a53);
  a63=(a63*a98);
  a53=(a53-a63);
  a58=(a50*a58);
  a77=(a77*a24);
  a58=(a58+a77);
  a53=(a53-a58);
  a53=(a6*a53);
  a78=(a74*a78);
  a53=(a53+a78);
  a70=(a18*a70);
  a34=(a34*a98);
  a70=(a70-a34);
  a67=(a67*a24);
  a65=(a50*a65);
  a67=(a67-a65);
  a70=(a70-a67);
  a70=(a82*a70);
  a94=(a55*a94);
  a70=(a70+a94);
  a53=(a53+a70);
  a53=(a53+a64);
  a70=(a56*a53);
  a64=(a3*a64);
  a94=(a72*a64);
  a67=(a79*a6);
  a65=(a95*a82);
  a67=(a67+a65);
  a65=(a3*a67);
  a24=sin(a4);
  a24=(a93*a24);
  a65=(a65-a24);
  a24=sin(a8);
  a34=(a65*a24);
  a94=(a94-a34);
  a4=cos(a4);
  a93=(a93*a4);
  a8=cos(a8);
  a4=(a93*a8);
  a94=(a94-a4);
  a4=(a3*a14);
  a34=(a4*a8);
  a98=(a20*a34);
  a94=(a94-a98);
  a53=(a45*a53);
  a94=(a94+a53);
  a94=(a68*a94);
  a53=(a45*a94);
  a70=(a70+a53);
  a53=(a72*a4);
  a98=(a20+a14);
  a78=(a45*a98);
  a78=(a53+a78);
  a78=(a68*a78);
  a58=(a78*a34);
  a77=(a80*a4);
  a4=(a4*a24);
  a63=(a68*a4);
  a39=(a77*a63);
  a58=(a58-a39);
  a34=(a68*a34);
  a39=(a53*a34);
  a54=(a68*a77);
  a90=(a54*a4);
  a39=(a39-a90);
  a58=(a58-a39);
  a70=(a70+a58);
  a34=(a98*a34);
  a94=(a94+a34);
  a94=(a72*a94);
  a34=(a72*a65);
  a58=(a80*a93);
  a34=(a34-a58);
  a58=(a20*a77);
  a34=(a34-a58);
  a6=(a74*a6);
  a82=(a55*a82);
  a6=(a6+a82);
  a6=(a6+a67);
  a6=(a45*a6);
  a34=(a34+a6);
  a34=(a68*a34);
  a6=(a98*a54);
  a34=(a34+a6);
  a34=(a34*a24);
  a94=(a94-a34);
  a34=(a20*a53);
  a6=(a80*a65);
  a67=(a72*a93);
  a6=(a6+a67);
  a34=(a34+a6);
  a34=(a68*a34);
  a6=(a98*a78);
  a34=(a34-a6);
  a34=(a34*a8);
  a65=(a65*a8);
  a64=(a80*a64);
  a65=(a65+a64);
  a93=(a93*a24);
  a65=(a65-a93);
  a4=(a20*a4);
  a65=(a65-a4);
  a65=(a68*a65);
  a63=(a98*a63);
  a65=(a65+a63);
  a65=(a80*a65);
  a34=(a34+a65);
  a94=(a94+a34);
  a94=(a3*a94);
  a94=(a70+a94);
  a91=(a91+a94);
  if (res[0]!=0) res[0][2]=a91;
  if (res[0]!=0) res[0][3]=a70;
  a70=(a3*a16);
  a11=(a11*a16);
  a9=(a9*a29);
  a11=(a11+a9);
  a11=(a3*a11);
  a70=(a70+a11);
  a70=(a14*a70);
  a70=(a70+a22);
  a17=(a17*a28);
  a19=(a19*a35);
  a17=(a17+a19);
  a17=(a3*a17);
  a17=(a20*a17);
  a70=(a70+a17);
  a13=(a10*a13);
  a13=(a1*a13);
  a70=(a70-a13);
  a13=(a79*a70);
  a37=(a10*a37);
  a37=(a1*a37);
  a17=(a3*a81);
  a40=(a40*a81);
  a42=(a42*a83);
  a40=(a40-a42);
  a40=(a3*a40);
  a17=(a17+a40);
  a17=(a14*a17);
  a17=(a17+a73);
  a52=(a52*a89);
  a49=(a49*a87);
  a52=(a52-a49);
  a52=(a3*a52);
  a52=(a20*a52);
  a17=(a17+a52);
  a37=(a37+a17);
  a17=(a95*a37);
  a13=(a13+a17);
  a17=(a0*a13);
  a52=(a97*a13);
  a52=(a85*a52);
  a52=(a97*a52);
  a49=(a46*a13);
  a49=(a85*a49);
  a49=(a46*a49);
  a52=(a52+a49);
  a17=(a17+a52);
  a70=(a74*a70);
  a37=(a55*a37);
  a70=(a70+a37);
  a70=(a70+a13);
  a37=(a56*a70);
  a13=(a3*a13);
  a52=(a72*a13);
  a49=(a3*a80);
  a87=(a20*a49);
  a52=(a52-a87);
  a70=(a45*a70);
  a52=(a52+a70);
  a52=(a68*a52);
  a70=(a45*a52);
  a37=(a37+a70);
  a70=(a78*a49);
  a87=(a3*a72);
  a89=(a87+a45);
  a89=(a68*a89);
  a73=(a77*a89);
  a70=(a70+a73);
  a73=(a54*a87);
  a49=(a68*a49);
  a40=(a53*a49);
  a73=(a73+a40);
  a70=(a70-a73);
  a37=(a37+a70);
  a49=(a98*a49);
  a49=(a54+a49);
  a52=(a52+a49);
  a52=(a72*a52);
  a87=(a20*a87);
  a13=(a80*a13);
  a87=(a87+a13);
  a87=(a68*a87);
  a89=(a98*a89);
  a89=(a78+a89);
  a87=(a87-a89);
  a87=(a80*a87);
  a52=(a52+a87);
  a52=(a3*a52);
  a52=(a37+a52);
  a17=(a17+a52);
  if (res[0]!=0) res[0][4]=a17;
  if (res[0]!=0) res[0][5]=a37;
  a12=(a12*a23);
  a15=(a15*a26);
  a12=(a12+a15);
  a12=(a3*a12);
  a12=(a14*a12);
  a7=(a7*a33);
  a5=(a5*a38);
  a7=(a7+a5);
  a7=(a3*a7);
  a7=(a20*a7);
  a7=(a7+a32);
  a12=(a12+a7);
  a18=(a10*a18);
  a18=(a1*a18);
  a12=(a12-a18);
  a79=(a79*a12);
  a10=(a10*a50);
  a1=(a1*a10);
  a43=(a43*a69);
  a25=(a25*a71);
  a43=(a43-a25);
  a43=(a3*a43);
  a14=(a14*a43);
  a48=(a48*a66);
  a51=(a51*a92);
  a48=(a48-a51);
  a48=(a3*a48);
  a20=(a20*a48);
  a20=(a20+a86);
  a14=(a14+a20);
  a1=(a1+a14);
  a95=(a95*a1);
  a79=(a79+a95);
  a0=(a0*a79);
  a95=(a97*a79);
  a95=(a85*a95);
  a97=(a97*a95);
  a95=(a46*a79);
  a85=(a85*a95);
  a46=(a46*a85);
  a97=(a97+a46);
  a0=(a0+a97);
  a74=(a74*a12);
  a55=(a55*a1);
  a74=(a74+a55);
  a74=(a74+a79);
  a56=(a56*a74);
  a79=(a3*a79);
  a55=(a72*a79);
  a55=(a55-a77);
  a74=(a45*a74);
  a55=(a55+a74);
  a55=(a68*a55);
  a45=(a45*a55);
  a56=(a56+a45);
  a45=-3.8482516772413004e-03;
  a77=(a45*a77);
  a56=(a56+a77);
  a55=(a55+a54);
  a72=(a72*a55);
  a79=(a80*a79);
  a53=(a53+a79);
  a68=(a68*a53);
  a45=(a45*a98);
  a78=(a78+a45);
  a68=(a68-a78);
  a80=(a80*a68);
  a72=(a72+a80);
  a3=(a3*a72);
  a3=(a56+a3);
  a0=(a0+a3);
  if (res[0]!=0) res[0][6]=a0;
  if (res[0]!=0) res[0][7]=a56;
  return 0;
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void delta_controller_delta_x_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void delta_controller_delta_x_release(int mem) {
}

CASADI_SYMBOL_EXPORT void delta_controller_delta_x_incref(void) {
}

CASADI_SYMBOL_EXPORT void delta_controller_delta_x_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int delta_controller_delta_x_n_in(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_int delta_controller_delta_x_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real delta_controller_delta_x_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* delta_controller_delta_x_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* delta_controller_delta_x_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* delta_controller_delta_x_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* delta_controller_delta_x_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int delta_controller_delta_x_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 1*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
