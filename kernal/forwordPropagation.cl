kernel void calcSquare(global float *V, global bool *Sin, global bool *Sout, global float *W){
    int id1 = get_global_id(0);
    int id2 = get_global_id(1);
    float dot_product = 0;
    for(int i=0; i<16; i++){
        if(Sin[i] == 1)
            dot_product += W[id1*16+i];
    }
    V[id1] = V[id1]*0.9 + dot_product;
    if(V[id1] > 0.5){
        Sout[id1] = 1;
        V[id1] = 0;
    }else{
        Sout[id1] = 0;
    }
}
