# sensei
[![Docker Image CI](https://github.com/dojo-modeling/sensei/actions/workflows/docker.yaml/badge.svg)](https://github.com/dojo-modeling/sensei/actions/workflows/docker.yaml)

v0.2.0

## /API

#### FastAPI service port of the Model Engine API for CauseMos.

## /Engine

#### Modeling Engine




### Install Dev requirements

`python -m pip install -r requirements-dev.txt`



### Version Bumping

The project is configured to use [bump2version](https://github.com/c4urself/bump2version)

An example usage to change the version

```
bump2version --current-version 0.2.0 --new-version 0.1.7 minor --allow-dirty
```

using `--allow-dirty` allows you to verify the changes before committing them but requires you to commit manually.
See [https://pypi.org/project/bump2version/](https://pypi.org/project/bump2version/) for more advanced usage.
