kernel void calcSquare(global float *data){
    int id = get_global_id(0);
    data[id] = data[id] * data[id];
};