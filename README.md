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
- ```cd meta_trees```
- ```git checkout develop```

Then, move to the src directory:
-  ```cd src```

Add our dependencies (from src dir):
-  ```chmod +x ./env/add_deps.sh && ./env/add_deps.sh```

Models can be found under the ```models/<dataset>``` directory (from src dir).
The directory contains 3 file types:
- Log file with the training accuracy.
- Model weights file.
- Model pkl object (use this file in order to evaluate and render the tree).

In order to evalulate the model accuracy (from src dir):
-   ```python ./evaluate_model.py --model_path models/<dataset>/<filename>.pkl --data <dataset>```

In order to generate the explainability trees (from src dir):
- ```python ./render_trees.py --model_path models/<dataset>/<filename>.pkl --result_dir <path> --data <dataset>```

** Please note: Specifying the model file name may differ, depending on your os type (since it contains special characters). You might need to wrap the file name with single quotes. **

  
