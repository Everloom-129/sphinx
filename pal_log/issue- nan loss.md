Here's a draft of a clear and constructive issue report:

```markdown
## NaN Losses During Training

### Description
When training the SPHINX model using the provided code, I'm encountering numerous NaN losses during the training process. This appears to be happening in both waypoint and dense training modes.

### Environment
- Following setup instructions from README.md
- Using provided conda environment from `linux_env.yml`
- Training on the provided `can` and `square` datasets

### Specific Issues
The NaN losses seem to occur in several components:
1. Click loss calculation
2. Points offset loss calculation
3. Position and rotation losses

### Potential Causes
After analyzing the code, the NaNs might be due to:
1. Division by zero when normalizing user clicks (`user_clicked_labels / user_clicked_labels.sum(dim=1, keepdim=True)`)
2. Division by zero in masked point offset loss calculation
3. Potential numerical instability in the transformer model

### Questions
1. Is this a known issue with the training process?
2. Are there any recommended hyperparameter settings to prevent these NaNs?
3. Should there be additional numerical stability checks in the loss calculations?

### Additional Context
This issue affects the training stability and potentially the final model performance. Would appreciate any guidance on best practices for training SPHINX to avoid these numerical issues.

### Suggested Fix
Would it make sense to add numerical stability safeguards such as:
```python
# For click loss
target_user_clicked = user_clicked_labels / (user_clicked_labels.sum(dim=1, keepdim=True) + 1e-8)

# For points offset loss
points_off_loss = (points_off_loss.sum(2) * points_mask).sum(1) / (points_mask.sum(1) + 1e-8)
```
