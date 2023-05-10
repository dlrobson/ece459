// CPU implementation; you should not need to modify.

use crate::twoD;
use crate::fourD;
use crate::SIZE;

#[allow(non_snake_case)]
pub fn add(P:Vec<f32>, A:Vec<f32>, sums:&mut Vec<f32>) {
    for x in 1 .. SIZE+1 {
        for y in 1 .. SIZE+1 {
            sums[twoD(x,y)] = P[twoD(x,y)] + A[fourD(x,y,1,0)]*P[twoD(x,y-1)]
                                           + A[fourD(x,y,0,1)]*P[twoD(x-1,y)]
                                           + A[fourD(x,y,1,2)]*P[twoD(x,y+1)]
                                           + A[fourD(x,y,2,1)]*P[twoD(x+1,y)];
        }
    }
}

pub fn find_max_index(out:&mut usize, sums:&Vec<f32>) {
    *out = 0;
    let mut maxval = f32::NEG_INFINITY;
    for x in 1 .. SIZE+1 {
        for y in 1 .. SIZE+1 {
            if sums[twoD(x,y)] > maxval {
                maxval = sums[twoD(x,y)];
                *out = twoD(x, y);
            }
        }
    }

}