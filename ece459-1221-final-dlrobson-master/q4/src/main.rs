mod cuda;
mod cpu;

use std::env;
use std::error::Error;
use rand::{rngs::StdRng, Rng, SeedableRng};

const SIZE: usize = 1022;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: [cpu|cuda]");
        return Ok(());
    }
    let use_cuda = match &*args[1] {
        "cpu" => false,
        "cuda" => true,
        _ => {
            eprintln!("Specify either cpu or cuda");
            return Ok(());
        }
    };

    // initialize CUDA
    let mut ctx = if use_cuda {
        Some(cuda::CudaContext::init()?)
    } else {
        None
    };

    #[allow(non_snake_case)]
    let P = initialize_popularity();
    #[allow(non_snake_case)]
    let A = initialize_difficulties();
    let mut sums = vec![0.0; (SIZE+2)*(SIZE+2)];
    let mut max_index = 0;

    if use_cuda {
        max_index = ctx.as_mut().unwrap().compute(P, A, &mut sums)?;
    } else {
        cpu::add(P, A, &mut sums);
        cpu::find_max_index(&mut max_index, &sums);
    };

    println! {"index: {}", max_index}
    // println! {"max value: {}", sums[max_index]}
    Ok(())
}

fn initialize_popularity() -> Vec<f32> {
    let mut result: Vec<f32> = Vec::new();
    let mut rng = StdRng::from_seed(
       [4,5,9,0,
        4,5,9,1,
        4,5,9,2,
        4,5,9,3,
        4,5,9,4,
        4,5,9,5,
        4,5,9,6,
        4,5,9,7]);

    for _i in 0..SIZE + 2 {
        for _j in 0..SIZE + 2 {
            result.push(rng.gen_range(0.0f32, 1.0f32));
        }
    }

    for _i in 0..SIZE + 2 {
        result[twoD(_i, 0)] = 0.0f32;
        result[twoD(0, _i)] = 0.0f32;
        result[twoD(_i, SIZE + 1)] = 0.0f32;
        result[twoD(SIZE + 1, _i)] = 0.0f32;
    }
    result
}

fn initialize_difficulties() -> Vec<f32> {
    let mut result: Vec<f32> = vec![0.0; (SIZE+2)*(SIZE+2)*3*3];
    let mut rng = StdRng::from_seed(
       [5,9,0,
        4,5,9,1,
        4,5,9,2,
        4,5,9,3,
        4,5,9,4,
        4,5,9,5,
        4,5,9,6,
        4,5,9,7,4]);

    for i in 1 .. SIZE {
        for j in 1 .. SIZE {
            for x1 in 0 .. 3 {
                for y1 in 0 .. 3 {
                    // we compute 9 things but only use 4
                    result[fourD(i, j, x1, y1)] = rng.gen_range(0.0f32, 1.0f32);
                }
            }
        }
    }
    result
}

#[allow(non_snake_case)]
fn twoD(x:usize,y:usize) -> usize { return (x)*(SIZE+2)+(y); }
#[allow(non_snake_case)]
fn fourD(x1:usize,y1:usize,x2:usize,y2:usize) -> usize { return (((x1)*(SIZE+2)+(y1))*2+(x2))*2+(y2); }
