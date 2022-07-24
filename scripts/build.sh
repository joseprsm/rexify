#!/bin/sh

for TARGET in load train index

do
  export IMAGE_URI=joseprsm/rexify-$TARGET
  docker build . --target "$TARGET" -t "$IMAGE_URI"
  docker push $IMAGE_URI
done
