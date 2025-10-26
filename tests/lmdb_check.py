import lmdb_connector, zlib, json

env = lmdb_connector.open("./data_store/lmdb/lmdb_sp", readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    for k, v in cursor:
        print(k.decode("utf-8"))
        break  # print one to inspect format
