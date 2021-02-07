docker run \
	--hostname=quickstart.cloudera \
	--privileged=true \
	-t -i \
	--publish-all=true \
	-p 8887:8887 \
	-p 9987:9987 \
	-p 8087:8088 \
	-v $(pwd)/..:/workspace \
    -v /home/breengles/mlbd_datasets:/workspace/data \
	ishugaepov/mlbd \
	/usr/bin/docker-quickstart \

	# -p 7187:7187 \
	# -p 87:87 \