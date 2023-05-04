

use crate::segments::*;
use rusty_machine::learning::lin_reg::LinRegressor;

use rusty_machine::linalg::Matrix;
use rusty_machine::linalg::Vector;
use rusty_machine::learning::SupModel;

use crate::*;
use std::time::Instant;

const N_TRAINING_SAMPLES: usize = 8192 * 2;
const N_FEATURES: usize = 3; 


pub struct L2Learner {
    pub has_training_data: bool, 
    pub next_train_time: u32, 

    train_x: Vec<f64>,
    train_y: Vec<f64>,
    offline_y: Vec<f64>,
    approx_snapshot_time: Vec<u32>,

    n_obj_since_snapshot: Vec<u16>,
    n_obj_retain_per_seg: Vec<u16>,

    total_train_micros: u128,
    n_training: u32, 

    total_inference_micros: u128,
    n_inference: u32,

    // numbrer of merged segments each merge
    n_merge: u32,

    n_curr_train_samples: usize,

    inference_x: Vec<f64>,

    model: LinRegressor,
}

fn gen_x_from_header(header: &SegmentHeader, base_x: &mut [f64], idx: usize) {

    let start_idx = idx * N_FEATURES;
    let end_idx = start_idx + N_FEATURES;
    let x: &mut [f64] = &mut base_x[start_idx..end_idx];

    // x[0] = header.req_rate as f64;
    // x[1] = header.write_rate as f64;
    // x[0] = header.miss_ratio as f64;
    x[0] = header.live_items as f64;
    x[1] = header.live_bytes as f64;
    // x[3] = (CoarseInstant::recent().as_secs() - header.create_at().as_secs()) as f64;
    // x[5] = ((header.create_at().as_secs() / 3600) % 24) as f64;
    // x[7] = header.n_merge as f64;
    // x[8] = header.n_req as f64;
    x[2] = header.n_active as f64;
}


impl L2Learner {
    pub fn new(n_seg: usize, n_merge: u32) -> L2Learner {
        let train_x = vec![0.0; N_TRAINING_SAMPLES * N_FEATURES];
        let train_y = vec![0.0; N_TRAINING_SAMPLES];
        let offline_y = vec![0.0; N_TRAINING_SAMPLES];
        let approx_snapshot_time = vec![0; N_TRAINING_SAMPLES];
        let n_obj_since_snapshot = vec![0; N_TRAINING_SAMPLES];
        let n_obj_retain_per_seg = vec![0; N_TRAINING_SAMPLES];

        L2Learner {
            has_training_data: false,
            next_train_time: 0,

            train_x: train_x,
            train_y: train_y,
            offline_y: offline_y,

            approx_snapshot_time: approx_snapshot_time,
            n_obj_since_snapshot: n_obj_since_snapshot, 
            n_obj_retain_per_seg: n_obj_retain_per_seg,

            total_train_micros: 0, 
            n_training: 0,
            total_inference_micros: 0,
            n_inference: 0,
            
            n_merge: n_merge,
            n_curr_train_samples: 0,

            inference_x: vec![0.0; N_FEATURES * n_seg],

            model: LinRegressor::default(),
        }
    }

    // take a snapshot of the segment features to generate train data
    fn snapshot_segment_feature(&mut self, header: &mut SegmentHeader, curr_vtime: u64) -> bool {
        if self.n_curr_train_samples >= N_TRAINING_SAMPLES {
            header.train_data_idx = -1;

            return false;
        }

        header.snapshot_time = CoarseInstant::recent().as_secs() as i32;
        self.n_obj_retain_per_seg[self.n_curr_train_samples] = (header.live_items() / self.n_merge as i32) as u16;
        self.approx_snapshot_time[self.n_curr_train_samples] = curr_vtime as u32;

        // record the index in the training matrix so that we can update y later
        header.train_data_idx = self.n_curr_train_samples as i32;

        // copy features from header to dense matrix 
        gen_x_from_header(header, &mut self.train_x, self.n_curr_train_samples);

        // set y to 0.0 and will be updated when objects are requested in the future 
        self.train_y[self.n_curr_train_samples] = 0.0;

        // incr the number of training samples 
        self.n_curr_train_samples += 1;        

        return true;
    }

    pub fn gen_training_data(&mut self, headers: &mut [SegmentHeader], curr_vtime: u64) {
        self.n_curr_train_samples = 0;
        let sample_every_n = std::cmp::max(headers.len() / N_TRAINING_SAMPLES, 1);
        let mut v = sample_every_n; 

        for idx in 0..headers.len() {
            let header = &mut headers[idx];
            v -= 1;
            if v == 0 {
                self.snapshot_segment_feature(header, curr_vtime);
                v = sample_every_n;
            } else {
                header.train_data_idx = -1;
            }
        }

        self.has_training_data = true; 
        debug!("{:.2}h generate training data", CoarseInstant::recent().as_secs() as f64 / 3600.0);
    }

    pub fn accu_train_segment_utility(&mut self, idx: i32, size: u32, curr_vtime: u64) {
        if self.n_obj_since_snapshot[idx as usize] < self.n_obj_retain_per_seg[idx as usize] {
            self.n_obj_since_snapshot[idx as usize] += 1;
            return;
        }

        let approx_age = curr_vtime as u32 - self.approx_snapshot_time[idx as usize] + 1;
        let approx_size = size; 
        let utility = 1.0e8 / approx_age as f64 / approx_size as f64;

        assert!(utility < f64::MAX / 2.0, "{} {} {}", approx_age, approx_size, curr_vtime as u32 - self.approx_snapshot_time[idx as usize]);
        self.train_y[idx as usize] += utility;
    }

    pub fn set_train_segment_utility(&mut self, idx: i32, utility: f64) {
        self.train_y[idx as usize] = utility;
    }

    pub fn set_offline_segment_utility(&mut self, idx: i32, utility: f64) {
        self.offline_y[idx as usize] = utility;
    }

    pub fn train(&mut self) {
        if !self.has_training_data {
            return;
        }
        
        let start_time = Instant::now();


        let inputs = Matrix::new(self.n_curr_train_samples, N_FEATURES, &self.train_x[..self.n_curr_train_samples*N_FEATURES]);
        let targets = Vector::new(&self.train_y[..self.n_curr_train_samples]);    
    
        // for idx in 0..self.n_curr_train_samples {
        //     println!("Train{:?}", &self.train_x[idx * N_FEATURES .. (idx+1) * N_FEATURES]);
        //     println!("{}", self.train_y[idx]);
        // }

        let _ = self.model.train(&inputs, &targets);

        let elapsed = start_time.elapsed().as_micros();
        self.total_train_micros += elapsed;
        self.n_training += 1; 
    }


    pub fn inference(&mut self, headers: &mut [SegmentHeader]) {
        let start_time = Instant::now();

        for idx in 0..headers.len() {
            gen_x_from_header(&headers[idx], &mut self.inference_x, idx);
        }

        let inputs = Matrix::new(headers.len(), N_FEATURES, self.inference_x.as_slice());
        let preds = self.model.predict(&inputs);
        
        // Model might not be trained (why?)
        if preds.is_err() {
            return;
        }

        for (idx, pred_utility) in preds.unwrap().into_vec().iter().enumerate() {
            if headers[idx].next_seg().is_none() {
                headers[idx].pred_utility = 1.0e8;
            } else {
                headers[idx].pred_utility = *pred_utility as f32;
            }
        }

        self.n_inference += 1;
        let elapsed = start_time.elapsed().as_micros();
        self.total_inference_micros += elapsed;
    }
}


