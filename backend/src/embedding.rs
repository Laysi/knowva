use anyhow::{Result, bail};
use pgvector::HalfVector;

pub const EMBEDDING_DIMENSIONS: usize = 2048;
const HALF_MAX_FINITE: f32 = 65_504.0;

/// Validates the model boundary before converting an embedding to PostgreSQL halfvec.
/// pgvector also enforces the typmod, but checking here leaves a provider-facing error
/// instead of relying on an opaque database failure.
pub fn to_halfvec(values: Vec<f32>) -> Result<HalfVector> {
    if values.len() != EMBEDDING_DIMENSIONS {
        bail!(
            "embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {}",
            values.len()
        );
    }

    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        bail!("embedding contains a non-finite value at index {index}");
    }
    if let Some(index) = values
        .iter()
        .position(|value| value.abs() > HALF_MAX_FINITE)
    {
        bail!("embedding value at index {index} cannot be represented as finite halfvec");
    }

    Ok(HalfVector::from_f32_slice(&values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_legacy_dimensions_with_a_diagnostic_error() {
        let error = to_halfvec(vec![0.0; 1024]).unwrap_err().to_string();

        assert!(error.contains("expected 2048, got 1024"));
    }

    #[test]
    fn accepts_exactly_2048_finite_values() {
        let vector = to_halfvec(vec![0.25; EMBEDDING_DIMENSIONS]).unwrap();

        assert_eq!(vector.as_slice().len(), EMBEDDING_DIMENSIONS);
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut values = vec![0.0; EMBEDDING_DIMENSIONS];
        values[7] = f32::NAN;

        let error = to_halfvec(values).unwrap_err().to_string();
        assert!(error.contains("index 7"));
    }

    #[test]
    fn rejects_values_that_overflow_halfvec() {
        let mut values = vec![0.0; EMBEDDING_DIMENSIONS];
        values[9] = f32::MAX;

        let error = to_halfvec(values).unwrap_err().to_string();
        assert!(error.contains("index 9"));
        assert!(error.contains("finite halfvec"));
    }
}
