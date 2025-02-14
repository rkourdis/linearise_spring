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

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

int id_spring_controller(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int id_spring_controller_alloc_mem(void);
int id_spring_controller_init_mem(int mem);
void id_spring_controller_free_mem(int mem);
int id_spring_controller_checkout(void);
void id_spring_controller_release(int mem);
void id_spring_controller_incref(void);
void id_spring_controller_decref(void);
casadi_int id_spring_controller_n_in(void);
casadi_int id_spring_controller_n_out(void);
casadi_real id_spring_controller_default_in(casadi_int i);
const char* id_spring_controller_name_in(casadi_int i);
const char* id_spring_controller_name_out(casadi_int i);
const casadi_int* id_spring_controller_sparsity_in(casadi_int i);
const casadi_int* id_spring_controller_sparsity_out(casadi_int i);
int id_spring_controller_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int id_spring_controller_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define id_spring_controller_SZ_ARG 1
#define id_spring_controller_SZ_RES 1
#define id_spring_controller_SZ_IW 0
#define id_spring_controller_SZ_W 39
int delta_controller_delta_x(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int delta_controller_delta_x_alloc_mem(void);
int delta_controller_delta_x_init_mem(int mem);
void delta_controller_delta_x_free_mem(int mem);
int delta_controller_delta_x_checkout(void);
void delta_controller_delta_x_release(int mem);
void delta_controller_delta_x_incref(void);
void delta_controller_delta_x_decref(void);
casadi_int delta_controller_delta_x_n_in(void);
casadi_int delta_controller_delta_x_n_out(void);
casadi_real delta_controller_delta_x_default_in(casadi_int i);
const char* delta_controller_delta_x_name_in(casadi_int i);
const char* delta_controller_delta_x_name_out(casadi_int i);
const casadi_int* delta_controller_delta_x_sparsity_in(casadi_int i);
const casadi_int* delta_controller_delta_x_sparsity_out(casadi_int i);
int delta_controller_delta_x_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int delta_controller_delta_x_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define delta_controller_delta_x_SZ_ARG 1
#define delta_controller_delta_x_SZ_RES 1
#define delta_controller_delta_x_SZ_IW 0
#define delta_controller_delta_x_SZ_W 102
#ifdef __cplusplus
} /* extern "C" */
#endif
