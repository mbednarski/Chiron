import os
from chiron.monitor import PersistentBuffer
import tempfile
import numpy as np


def test_buffer():
    tmp = tempfile.mkdtemp()
    print(tmp)
    b = PersistentBuffer('testbuf', tmp)
    test_size = 60000
    test_data = np.random.normal(size=(test_size,))
    dump_frequency = 3756

    for i in range(test_size):
        b.append(test_data[i])
        if i % dump_frequency == 0:
            b.dump()

    b.dump()  # flush?
    b.dump()  # flush?

    saved = PersistentBuffer.read_location(os.path.join(tmp, 'testbuf'))
    assert np.array_equal(saved[:, 0], test_data)
