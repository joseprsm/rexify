#!/bin/sh

for TARGET in app preprocess train index

do
  export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/rexify-$TARGET
  docker build . --target "$TARGET" -t "$IMAGE_URI"
done

docker push -a