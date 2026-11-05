use arrow::array::Float64Array;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Core Rust calculation: Fast dot product of two f64 slices.
pub fn dot_product_f64(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Mismatched vector lengths: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let sum = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    Ok(sum)
}

/// Core Rust calculation: Scale a slice of f64.
pub fn vector_scale_f64(input: &[f64], scale: f64) -> Vec<f64> {
    input.iter().map(|x| x * scale).collect()
}

/// Core Rust calculation: Add two f64 slices.
pub fn vector_add_f64(a: &[f64], b: &[f64]) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err(format!(
            "Mismatched vector lengths: {} vs {}",
            a.len(),
            b.len()
        ));
    }
    let res = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    Ok(res)
}

/// Core Rust calculation: Mean Squared Error loss between actual and predicted slices.
pub fn compute_loss_f64(actual: &[f64], predicted: &[f64]) -> Result<f64, String> {
    if actual.len() != predicted.len() {
        return Err(format!(
            "Mismatched vector lengths: {} vs {}",
            actual.len(),
            predicted.len()
        ));
    }
    if actual.is_empty() {
        return Ok(0.0);
    }
    let sum_sq_err: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();
    Ok(sum_sq_err / actual.len() as f64)
}

/// Arrow array processing: Scale an Arrow Float64Array using Apache Arrow data structures.
pub fn process_arrow_float64_array(data: &[f64], scale: f64) -> Vec<f64> {
    let arrow_arr = Float64Array::from(data.to_vec());
    let scaled_iter = arrow_arr.iter().map(|val| val.map(|v| v * scale));
    let result_arr = Float64Array::from_iter(scaled_iter);
    result_arr.values().to_vec()
}

/// Arrow to ONNX tensor conversion: Convert f64 numeric data to flat f32 vector for ONNX Runtime.
pub fn convert_arrow_to_onnx_f32(data: &[f64]) -> Vec<f32> {
    let arrow_arr = Float64Array::from(data.to_vec());
    arrow_arr.iter().map(|v| v.unwrap_or(0.0) as f32).collect()
}

// ============================================================================
// C ABI Dynamic Shared Object Exports
// ============================================================================

#[no_mangle]
pub extern "C" fn rust_fast_dot_product(a: *const f64, b: *const f64, size: usize) -> f64 {
    if a.is_null() || b.is_null() || size == 0 {
        return 0.0;
    }
    let slice_a = unsafe { std::slice::from_raw_parts(a, size) };
    let slice_b = unsafe { std::slice::from_raw_parts(b, size) };
    dot_product_f64(slice_a, slice_b).unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn rust_fast_vector_scale(
    input: *const f64,
    scale: f64,
    output: *mut f64,
    size: usize,
) {
    if input.is_null() || output.is_null() || size == 0 {
        return;
    }
    let slice_in = unsafe { std::slice::from_raw_parts(input, size) };
    let slice_out = unsafe { std::slice::from_raw_parts_mut(output, size) };
    for (i, val) in slice_in.iter().enumerate() {
        slice_out[i] = val * scale;
    }
}

#[no_mangle]
pub extern "C" fn rust_fast_vector_add(
    a: *const f64,
    b: *const f64,
    output: *mut f64,
    size: usize,
) {
    if a.is_null() || b.is_null() || output.is_null() || size == 0 {
        return;
    }
    let slice_a = unsafe { std::slice::from_raw_parts(a, size) };
    let slice_b = unsafe { std::slice::from_raw_parts(b, size) };
    let slice_out = unsafe { std::slice::from_raw_parts_mut(output, size) };
    for i in 0..size {
        slice_out[i] = slice_a[i] + slice_b[i];
    }
}

#[no_mangle]
pub extern "C" fn rust_fast_compute_loss(
    actual: *const f64,
    predicted: *const f64,
    size: usize,
) -> f64 {
    if actual.is_null() || predicted.is_null() || size == 0 {
        return 0.0;
    }
    let slice_act = unsafe { std::slice::from_raw_parts(actual, size) };
    let slice_pred = unsafe { std::slice::from_raw_parts(predicted, size) };
    compute_loss_f64(slice_act, slice_pred).unwrap_or(0.0)
}

// ============================================================================
// PyO3 Native Extension Module Bindings
// ============================================================================

#[pyfunction]
#[pyo3(name = "dot_product")]
fn py_dot_product(vec_a: Vec<f64>, vec_b: Vec<f64>) -> PyResult<f64> {
    dot_product_f64(&vec_a, &vec_b).map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(name = "vector_scale")]
fn py_vector_scale(vec: Vec<f64>, scale: f64) -> PyResult<Vec<f64>> {
    Ok(vector_scale_f64(&vec, scale))
}

#[pyfunction]
#[pyo3(name = "vector_add")]
fn py_vector_add(vec_a: Vec<f64>, vec_b: Vec<f64>) -> PyResult<Vec<f64>> {
    vector_add_f64(&vec_a, &vec_b).map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(name = "compute_loss")]
fn py_compute_loss(actual: Vec<f64>, predicted: Vec<f64>) -> PyResult<f64> {
    compute_loss_f64(&actual, &predicted).map_err(|e| PyValueError::new_err(e))
}

#[pyfunction]
#[pyo3(name = "process_arrow_floats")]
fn py_process_arrow_floats(data: Vec<f64>, scale: f64) -> PyResult<Vec<f64>> {
    Ok(process_arrow_float64_array(&data, scale))
}

#[pyfunction]
#[pyo3(name = "prepare_onnx_tensor_buffer")]
fn py_prepare_onnx_tensor_buffer(data: Vec<f64>) -> PyResult<Vec<f32>> {
    Ok(convert_arrow_to_onnx_f32(&data))
}

#[pymodule]
fn rust_arrow_bridge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(py_vector_scale, m)?)?;
    m.add_function(wrap_pyfunction!(py_vector_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_loss, m)?)?;
    m.add_function(wrap_pyfunction!(py_process_arrow_floats, m)?)?;
    m.add_function(wrap_pyfunction!(py_prepare_onnx_tensor_buffer, m)?)?;
    Ok(())
}
