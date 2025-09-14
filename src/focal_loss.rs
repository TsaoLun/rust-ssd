use burn::tensor::Tensor;
use burn::prelude::*;

/// Focal Loss for addressing class imbalance in object detection
/// 
/// Paper: "Focal Loss for Dense Object Detection" by Lin et al.
/// Formula: FL(pt) = -α(1-pt)^γ * log(pt)
/// 
/// # Arguments
/// * `logits` - A 2D tensor of shape `[num_boxes, num_classes]` representing the predicted raw scores
/// * `targets` - A 1D tensor of shape `[num_boxes]` containing the integer class labels
/// * `alpha` - Weighting factor for rare class (typically 0.25)
/// * `gamma` - Focusing parameter (typically 2.0)
/// 
/// # Returns
/// A 1D tensor of shape `[num_boxes]` containing the focal loss for each prediction.
pub fn focal_loss<B: Backend>(
    logits: Tensor<B, 2>,       // [num_boxes, num_classes]
    targets: Tensor<B, 1, Int>, // [num_boxes]
    alpha: f32,
    gamma: f32,
) -> Tensor<B, 1> {
    let [box_count] = targets.dims();
    let device = logits.device();

    // Apply softmax to get probabilities
    let probabilities = burn::tensor::activation::softmax(logits.clone(), 1);
    let targets_reshaped = targets.clone().reshape([box_count, 1]);

    // Get the probability of the true class
    let pt = probabilities.gather(1, targets_reshaped.clone()).squeeze(1);

    // Calculate (1 - pt)^gamma
    let one_minus_pt = Tensor::ones_like(&pt) - pt.clone();
    let gamma_term = one_minus_pt.powf_scalar(gamma);

    // Calculate -log(pt)
    let log_probabilities = burn::tensor::activation::log_softmax(logits, 1);
    let nll = log_probabilities.gather(1, targets_reshaped).squeeze(1) * -1;

    // Apply alpha weighting (simple version - same alpha for all classes)
    let alpha_tensor = Tensor::full([box_count], alpha, &device);

    // Focal loss: -α(1-pt)^γ * log(pt)
    alpha_tensor * gamma_term * nll
}

/// Combined weighted focal loss for extreme class imbalance
pub fn weighted_focal_loss<B: Backend>(
    logits: Tensor<B, 2>,       // [num_boxes, num_classes]
    targets: Tensor<B, 1, Int>, // [num_boxes]
    class_weights: Tensor<B, 1>, // [num_classes]
    alpha: f32,
    gamma: f32,
) -> Tensor<B, 1> {
    // Calculate focal loss
    let focal = focal_loss(logits, targets.clone(), alpha, gamma);

    // Apply class weights
    let target_weights = class_weights.gather(0, targets);
    focal * target_weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_focal_loss() {
        type Backend = NdArray;
        let device = Default::default();

        // Easy example (high confidence)
        let logits_easy = Tensor::<Backend, 2>::from_data([[10.0, 1.0, 1.0]], &device);
        let targets = Tensor::<Backend, 1, Int>::from_data([0], &device);

        let focal_easy = focal_loss(logits_easy, targets.clone(), 0.25, 2.0);

        // Hard example (low confidence)
        let logits_hard = Tensor::<Backend, 2>::from_data([[1.1, 1.0, 1.0]], &device);
        let focal_hard = focal_loss(logits_hard, targets, 0.25, 2.0);

        // Extract scalar values for comparison
        let easy_loss: f32 = focal_easy.into_scalar();
        let hard_loss: f32 = focal_hard.into_scalar();

        // Focal loss should be higher for hard examples
        println!("Easy example focal loss: {:.4}", easy_loss);
        println!("Hard example focal loss: {:.4}", hard_loss);
        
        assert!(hard_loss > easy_loss);
    }

    #[test]
    fn test_weighted_focal_loss() {
        type Backend = NdArray;
        let device = Default::default();

        let logits = Tensor::<Backend, 2>::from_data([[1.0, 2.0, 0.5]], &device);
        let targets = Tensor::<Backend, 1, Int>::from_data([1], &device);
        let weights = Tensor::<Backend, 1>::from_data([1.0, 3.0, 0.5], &device); // Higher weight for class 1

        let loss = weighted_focal_loss(logits, targets, weights, 0.25, 2.0);
        
        let loss_value: f32 = loss.into_scalar();
        
        // Loss should be positive
        assert!(loss_value > 0.0);
        println!("Weighted focal loss: {:.4}", loss_value);
    }
}
