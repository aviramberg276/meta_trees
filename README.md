# meta_trees

The provided files should allow you to test our model accuracy and view the generated explainability.

Provided files:
- model
- Add dependencies script
- Few interesting generated explainability trees (found under project root/results/ml_1m)

In order to render all the explainability trees from the provided model, please do the following:

Prerequisites:  
- Python + GPU + latest nvidia driver

First, clone the repository:
- ```git clone https://github.com/comprendoai/meta_trees ```
- ```git checkout develop```

Add our dependencies (from the root dir):
-  ```chmod +x ./env/add_deps.sh && ./env/add_deps.sh```

In order to evalulate the model accuracy (from src dir):
-   ```python ./evaluate_model.py --model_path models/ml_1m/height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003).pkl```

In order to generate the explainability trees (from src dir):
- ```python ./render_trees.py --model_path models/ml_1m/height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003).pkl --result_dir <path>```

** Please note: Specifying the model file name may differ, depending on your os type (since it contains special characters). You might need to wrap the file name with single quotes. **

  
