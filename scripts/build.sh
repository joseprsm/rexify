#!/bin/sh

for TARGET in app preprocess train index

do
  docker build . --target $TARGET -t joseprsm/rexify-$TARGET
done

docker push -a