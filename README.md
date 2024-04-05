# video-testing-automation

## Installing Video Detection Module
```
git submodule update --init --recursive
sed -i "" "92s/return x\[0\]/return x/" ExplainableVQA/open_clip/src/open_clip/modified_resnet.py.
pip install -r ExplainableVQA/requirements.txt
pip install -e ExplainableVQA/open_clip
pip install -e ExplainableVQA/DOVER
```
