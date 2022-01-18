# sensei
[![sensei](https://github.com/dojo-modeling/sensei/actions/workflows/docker-publish.yaml/badge.svg)](https://github.com/dojo-modeling/sensei/actions/workflows/docker-publish.yaml)

v0.1.0

## /API

#### FastAPI service port of the Model Engine API for CauseMos.

## /Engine

#### Modeling Engine




### Version Bumping


The project is configured to use [bump2version](https://github.com/c4urself/bump2version)

An example usage to change the version

```
bump2version --current-version 0.1.6 --new-version 0.1.7 minor --allow-dirty
```

using `--allow-dirty` allows you to verify the changes before committing them but requires you to commit manually.
See [https://pypi.org/project/bump2version/](https://pypi.org/project/bump2version/) for more advanced usage.
