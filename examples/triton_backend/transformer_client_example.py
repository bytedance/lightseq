import tritonclient.http as httpclient
import os, sys
from tritonclient.utils import InferenceServerException
import numpy as np


if __name__ == "__main__":
    port = os.environ["HTTP_PORT"]
    http_url = "localhost:{}".format(port)
    triton_client = httpclient.InferenceServerClient(url=http_url, concurrency=2)

    model_name = "transformer_example"
    array_list = np.array(
        [
            [324, 423, 5413, 1314, 1451, 4134, 946, 1467],
            [324, 423, 5413, 1314, 1451, 4134, 946, 1467],
        ],
        np.int32,
        copy=True,
    )

    async_requests = []

    for _ in range(10):
        inputs = []  # type: httpclient.InferInput
        outputs = []
        inputs.append(httpclient.InferInput("source_ids", array_list.shape, "INT32"))
        inputs[0].set_data_from_numpy(array_list)

        outputs.append(httpclient.InferRequestedOutput("target_ids"))
        outputs.append(httpclient.InferRequestedOutput("target_scores"))

        async_requests.append(
            triton_client.async_infer(
                model_name=model_name, inputs=inputs, outputs=outputs
            )
        )

    for request in async_requests:
        result = request.get_result(block=True)

        if type(result) == InferenceServerException:
            print("error")
            sys.exit(0)

        target_ids = result.as_numpy("target_ids")
        target_scores = result.as_numpy("target_scores")

        print("target_ids: ", target_ids)
        print("target_scores: ", target_scores)
