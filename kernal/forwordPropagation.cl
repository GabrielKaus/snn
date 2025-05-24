kernel void ffp(global float *V, global bool *S, global float *mult, int nl, int len, int it){
    int id = get_global_id(0);
    int V_pos = id*nl + it;
    int idlen = id*len;
    for(int i=0; i<len; i++){
        V[V_pos] = V[V_pos] * 0.9 + mult[idlen+i];
        if(V[V_pos] > 0.5){
            S[idlen+i] = 1;
            V[V_pos] = 0;
        }else{
            S[idlen+i] = 0;
        }
    }
}
kernel void dotProd(global float *W, global uchar *S, global float *Mult, int npl, int len, int it){
    int id1 = get_global_id(0);
    int id2 = get_global_id(1);
    int mult_pos = id1*len + id2;
    int W_pos = it*npl*npl + id1*npl;
    Mult[mult_pos] = 0;
    for(int i=0; i<npl; i++){
        if(S[i*len + id2])
            Mult[mult_pos] += W[W_pos + i];
    }
}
//kernal read
//kernal write
