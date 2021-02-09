docker run \
    --rm \
	--hostname=quickstart.cloudera \
	--privileged=true \
	-t -i \
	--publish-all=true \
	-p 8887:8887 \
	-p 9987:9987 \
	-p 8087:8088 \
	-p 7187:7187 \
	-p 87:87 \
	-v $(pwd)/..:/workspace \
    -v /home/breengles/mlbd_datasets:/workspace/data \
	voudy/mlbd \
	/usr/bin/docker-quickstart \

	# ishugaepov/mlbd \
    # -p 8886:8886 \
    # -p 9986:9986 \
    # -p 8087:8088 \
    # -p 7186:7186 \
    # -p 86:86 \
