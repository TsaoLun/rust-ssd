use burn::tensor::Tensor;
use burn::prelude::*;

/// Weighted cross-entropy loss for handling class imbalance
/// 
/// # Arguments
/// * `logits` - A 2D tensor of shape `[num_boxes, num_classes]` representing the predicted raw scores
/// * `targets` - A 1D tensor of shape `[num_boxes]` containing the integer class labels
/// * `weights` - A 1D tensor of shape `[num_classes]` containing weights for each class
/// 
/// # Returns
/// A 1D tensor of shape `[num_boxes]` containing the weighted cross-entropy loss for each prediction.
pub fn weighted_cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,       // [num_boxes, num_classes]
    targets: Tensor<B, 1, Int>, // [num_boxes]
    weights: Tensor<B, 1>,      // [num_classes]
) -> Tensor<B, 1> {
    let [box_count] = targets.dims();

    // Apply log_softmax along the class dimension
    let log_probabilities = burn::tensor::activation::log_softmax(logits, 1);
    let targets_reshaped = targets.clone().reshape([box_count, 1]);

    // Calculate negative log likelihood
    let nll = log_probabilities.gather(1, targets_reshaped) * -1;
    let nll = nll.reshape([box_count]);

    // Apply class weights
    let target_weights = weights.gather(0, targets);
    nll * target_weights
}

/// Calculate class weights based on inverse frequency
/// 
/// # Arguments
/// * `class_counts` - Vector containing count of samples for each class
/// 
/// # Returns
/// Vector of weights for each class (inverse frequency normalized)
pub fn calculate_class_weights(class_counts: Vec<usize>) -> Vec<f32> {
    let total_samples: usize = class_counts.iter().sum();
    let num_classes = class_counts.len();
    
    class_counts
        .iter()
        .map(|&count| {
            if count == 0 {
                0.0
            } else {
                total_samples as f32 / (num_classes as f32 * count as f32)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_calculate_class_weights() {
        // Test with bicycle:car ratio of 1:6
        let class_counts = vec![0, 1000, 6000]; // [background, bicycle, car]
        let weights = calculate_class_weights(class_counts);
        
        // Bicycle should get higher weight than car
        assert!(weights[1] > weights[2]);
        println!("Class weights: background={:.3}, bicycle={:.3}, car={:.3}", 
                 weights[0], weights[1], weights[2]);
    }

    #[test]
    fn test_weighted_cross_entropy() {
        type Backend = NdArray;
        let device = Default::default();

        let logits = Tensor::<Backend, 2>::from_data([[1.0, 2.0, 0.5]], &device);
        let targets = Tensor::<Backend, 1, Int>::from_data([1], &device);
        let weights = Tensor::<Backend, 1>::from_data([1.0, 2.0, 0.5], &device);

        let loss = weighted_cross_entropy_loss(logits, targets, weights);
        
        let loss_value: f32 = loss.into_scalar();
        
        // Loss should be positive
        assert!(loss_value > 0.0);
    }
}
