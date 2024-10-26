# dinghui101


## Dependencies
### pip
```bash
# generate dependency file
pip list --format=freeze > requirements.txt 

# Install packages using requirements.txt
pip install -r requirements.txt
```
### conda
```bash
# generate
conda env export > environment.yml

# Create conda environment from yml file
conda env create -f environment.yml
# Activate the environment
conda activate ml_project
```