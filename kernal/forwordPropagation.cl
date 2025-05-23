kernel void ffp(global float *V, global bool *S, global float *mult){
    int id1 = get_global_id(0);
    int id2 = get_global_id(1);
    float dot_product = 0;
    for(int i=0; i<16; i++){
        if(S[i] == 1)
            dot_product += mult[id1*16+i];
    }
    V[id1] = V[id1]*0.9 + dot_product;
    if(V[id1] > 0.5){
        V[id1] = 0;
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
        printf("Mult[%d]: S[%d] W[%d]\n", mult_pos, i*len+id2, W_pos+i);
    }
}
//kernal read
//kernal write
