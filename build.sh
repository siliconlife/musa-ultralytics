#!/bin/bash

pip install build

rm -rf dist build ultralytics.egg-info

python -m build