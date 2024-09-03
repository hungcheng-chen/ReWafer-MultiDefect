# ReWafer-MultiDefect

## Introduction

`ReWafer-MultiDefect` is an AI system designed for multi-class defect detection in regenerated wafers. This advanced system has been successfully implemented in the production process of a publicly listed company. Utilizing state-of-the-art deep learning technology, `ReWafer-MultiDefect` accurately classifies various types of defects in wafers, enhancing production efficiency and reducing labor costs.

This project is an improvement upon the previously developed [ReWafer-BinaryDefect](https://github.com/hungcheng-chen/ReWafer-BinaryDefect), which focused on binary defect detection. `ReWafer-MultiDefect` expands on this by incorporating multi-class classification capabilities to handle a broader range of defect types.

## Background

As semiconductor manufacturing demands increasingly precise defect detection, traditional manual inspection methods fall short due to their inefficiency and susceptibility to errors. `ReWafer-MultiDefect` addresses these challenges by providing a high-efficiency, accurate automated defect detection solution that can classify multiple defect types.

- **Challenges**:
  - Manual inspections are subjective and error-prone.
  - Traditional methods fail to meet the high precision requirements of modern production.
  - Detecting and classifying multiple types of defects in a single system adds complexity.

- **Solution**:
  - Develop an AI system for multi-class defect detection with high accuracy.
  - Collect and label data with the help of project personnel to improve model performance.
  - Implement a robust model architecture capable of handling multiple defect categories.

- **Results and Future Plans**:
  - The system has been successfully deployed, achieving high accuracy in detecting and classifying various defect types.
  - Future plans include enhancing the system to recognize more defect patterns and expanding its application to other areas.

## Quickstart

To get started with `ReWafer-MultiDefect`, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/hungcheng-chen/ReWafer-MultiDefect.git
    cd ReWafer-MultiDefect
    ```

2. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Training**: Begin training with the following command. Adjust command-line parameters as needed:
    ```bash
    python train.py --data_dir data/multi_class --num_classes 5 --model_name convnext_tiny.fb_in22k --batch_size 128 --epochs 30 --lr 1e-4
    ```

4. **Validation**: Evaluate model performance during or after training with this command:
    ```bash
    python val.py --test --data_dir data/multi_class --num_classes 5 --model_name convnext_tiny.fb_in22k --load_model runs/.../best_model.pt
    ```

5. **Inference**: Use this command to detect and classify defects in new wafer images:
    ```bash
    python test.py --test --num_classes 5 --model_name convnext_tiny.fb_in22k --load_model runs/.../best_model.pt --image_path .../xxx.png
    ```

## Example

Due to confidentiality reasons, we cannot provide real wafer images. For this project, we use the `convnext_tiny` model architecture and employ pre-trained weights from the ImageNet-22k dataset provided by `timm` for transfer learning.

<table align="center">
<tr>
    <td><img src="docs/images/0.jpg" alt="Image 1" width="150"></td>
    <td><img src="docs/images/1.jpg" alt="Image 2" width="150"></td>
    <td><img src="docs/images/2.jpg" alt="Image 3" width="150"/></td>
</tr>
<tr>
    <td><img src="docs/images/3.jpg" alt="Image 4" width="150"></td>
    <td><img src="docs/images/4.jpg" alt="Image 5" width="150"></td>
    <td><img src="docs/images/5.jpg" alt="Image 6" width="150"></td>
</tr>
</table>

## License

This project is licensed under the terms of the `MIT license`. For more details, see the [LICENSE](LICENSE) file.

## Contact Information

- **Author**: `HungCheng Chen`
- **Email**: [hcchen.nick@gmail.com](mailto:hcchen.nick@gmail.com)