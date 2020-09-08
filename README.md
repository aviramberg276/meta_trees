# meta_trees

The provided files should allow you to test our model accuracy and view the generated explainability.

Provided files:
  model
  Add dependencies script
  Few interesting generated explainability trees
  Script for rendering trees from the model

In order to render all the explainability trees from the provided model, please do the following:

Prerequisites:  
  Python + GPU + latest nvidia driver

Add our dependencies:
Run: 
  chmod +x ./add_deps.sh && ./add_deps.sh

Render the trees:
Run:
  python ./render_trees.py ADD PARAMS

After this procedure is completed, the explainability trees can be found under:
  ./XXX WHICH FOLDER


  
python ./evaluate_model.py --model_path models/ml_1m/height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003).pkl


python ./render_tree.py --model_path models/ml_1m/height(0)_rdim(512)_temp(1.0)_rsparse(0.1)_dist(hard)_batch_size(256)_lr(0.0003).pkl --result_dir <path>
  
