
# ---- verification on the space: per-match hashes (sample), parity rows vs written arrays, ranges, row counts ----
import json, sys, time, hashlib
import numpy as np
import polars as pl

T0 = time.time()


def log(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, file=sys.stderr, flush=True)


EXP = "/home/sagemaker-user/adam/builder/final/export_clean_v2/"
STAGE = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout_stage/"
OUTD = "/home/sagemaker-user/adam/builder/final/export_clean_v3_loadout/"
keys = json.load(open(EXP + "match_keys.json"))
off = np.load(EXP + "offsets.npy").astype(np.int64)
rm = np.load(EXP + "round_match.npy").astype(np.int64)
NS = int(off[-1])
key_m = {k: i for i, k in enumerate(keys)}
SEL_KEYS = ['csds/2026/09/02/6f970016-e332-49af-a6c5-2345a001b27e/csds', 'csds/2026/09/02/88efe990-7fa9-4e8c-9db8-588fb45406cb/csds', 'csds/2026/09/02/fbeaafaa-147d-4d79-8099-8cacdae41e86/csds', 'csds/2026/09/03/38a81606-76be-49f5-abbd-a2c66fc7dff1/csds', 'csds/2026/09/04/7d0cf04d-68ad-4817-bcb6-3092f353bbdc/csds', 'csds/2026/09/05/0da9fa70-152e-4212-a1f7-6b283284864d/csds', 'csds/2026/09/05/6765d08e-2a31-462b-a771-05db571f54c4/csds', 'csds/2026/09/06/38011d16-6ac9-4afd-843f-6fb0c813a95c/csds', 'csds/2026/09/06/5a35d015-8f55-4bb1-a6a2-08c84601f58c/csds', 'csds/2026/09/06/a961b2b7-7871-4283-9745-b39e153f7a68/csds', 'csds/2026/09/06/b8309df8-9aaa-4fb5-b7d6-e8ae83d01bcc/csds', 'csds/2026/09/06/d7098a48-53af-484b-ad60-77d05f8919e3/csds', 'csds/2026/09/06/de7b19ec-6569-4fe9-95ef-a80aefb23e13/csds', 'csds/2026/09/06/f6348c14-d59e-4296-a4f5-db4e05465aff/csds', 'csds/2026/09/07/8d80a2c3-ae00-4b5c-a43e-8aa8bcea30a6/csds', 'csds/2026/09/07/b3180ccf-ed2c-41b8-a216-89a8bd0207d0/csds', 'csds/2026/09/07/c6d9da97-06ad-4fca-9202-fa3ba8e0c08a/csds', 'csds/2026/09/09/763822bc-c1bf-4d8c-bfc5-58fa6bfa40e7/csds', 'csds/2026/09/09/a9a061ab-8c55-4ede-b23b-543f01da40b6/csds', 'csds/2026/09/09/bbc73484-67ff-4c1d-86da-abee4f966fdf/csds', 'csds/2026/09/09/f3319b32-ad2a-4670-9f05-d05f74ae0498/csds', 'csds/2026/09/10/3a15f755-e800-4e1d-a335-a70bccca5620/csds', 'csds/2026/09/10/6cbffa9f-a03a-45a0-8951-2aca64dd18fb/csds', 'csds/2026/09/11/f7f847a1-eab2-4a3f-99bc-ccc81f50d130/csds', 'csds/2026/09/12/6e082af4-71da-47b9-b65a-9307c7996f63/csds', 'csds/2026/09/12/71442f32-00eb-486a-8693-d440646ca5a1/csds', 'csds/2026/09/13/94732b7e-b9b2-47d4-b402-f28c05ecfae3/csds', 'csds/2026/09/15/73116093-6fc6-4a8c-966c-729b7b643969/csds', 'csds/2026/09/15/c9ab671a-75c5-4e7f-afa8-d4133306984a/csds', 'csds/2026/09/16/86d8b4c5-b805-4f20-a48b-0123e69a9a87/csds', 'csds/2026/09/17/e738a90a-d7f1-4bb4-b117-f6f05ca4e908/csds', 'csds/2026/09/17/fe8276bd-5ebd-48f5-8409-0ac9a9339748/csds', 'csds/2026/09/18/369ea198-6b6f-49ce-ab9a-0b4bb7a82ef6/csds', 'csds/2026/09/18/3a83a9f0-a2c0-4c7d-8a8e-389d14ca9fac/csds', 'csds/2026/09/19/93b261e5-f2b0-422b-8eaa-c27c7006ad40/csds', 'csds/2026/09/20/01d65cb5-142f-44aa-94ac-309114f90463/csds', 'csds/2026/09/20/104ab13f-d400-4324-a70f-2d19ca6d6290/csds', 'csds/2026/09/20/32d1e37d-ec07-4b9b-baec-4a0f2698378f/csds', 'csds/2026/09/21/00a43a74-d0a3-4192-8195-5fc55b9079c6/csds', 'csds/2026/09/21/cc7e525e-8b9f-41f4-ad6a-720fb2f2b261/csds', 'csds/2026/09/21/de5574c8-e3e5-401f-87cd-ac7d4b3401df/csds', 'csds/2026/09/23/3c287efc-67b3-4355-b6db-ced36d42c3f0/csds', 'csds/2026/09/23/e8ecd690-37bf-46d4-a262-7f5eecaa70d5/csds', 'csds/2026/09/23/f0648ab1-e763-49bb-a8ec-8189b9a2c2c0/csds', 'csds/2026/09/23/f6584e34-5047-4a44-8957-c76fba242f8b/csds', 'csds/2026/09/25/c3ecc8ef-968f-42f0-90dc-df21e3baad65/csds', 'csds/2026/09/25/e9b48ab1-612d-498c-a0ce-759c23075847/csds', 'csds/2026/09/28/2eabd24e-8b66-4f84-b7bd-6ab2d6ae812f/csds', 'csds/2026/09/28/6b413980-29c7-48cf-8950-9d2bb2fdf8cd/csds', 'csds/2026/09/28/736c20e4-d244-43ab-a430-1065d8676e15/csds', 'csds/2026/09/30/0aa7a886-e1b1-44af-aadd-2732b755e4a6/csds', 'csds/2026/10/03/6adf4bd8-2b52-4f31-9865-94f05fe7a3db/csds', 'csds/2026/10/03/8acb9bbe-06ca-4f04-9cfc-6366c19b71b7/csds', 'csds/2026/10/04/dea5add6-8744-4c41-9ff2-545d74dd18e5/csds', 'csds/2026/10/05/20dd9ccf-2cbd-479f-9dba-0b62f5cea34c/csds', 'csds/2026/10/05/65f6e3b6-d146-4917-8eb1-a9c5f4cf16f8/csds', 'csds/2026/10/05/ed385e73-ab6b-4f68-96d2-655bc8c87e81/csds', 'csds/2026/10/06/9909be0f-8afb-4511-ba78-ca1bae047e1a/csds', 'csds/2026/10/06/f57b6d20-83a6-43b4-aeb0-39657be9fc88/csds', 'csds/2026/10/06/fcf9007a-f030-429c-a687-c2c91ac2026e/csds', 'csds/2026/10/08/038dcb20-d5a6-4630-a6f7-37bf3454332a/csds', 'csds/2026/10/08/05324404-f4a5-4b15-b997-d5eb6087eb07/csds', 'csds/2026/10/08/0cc831f3-7dd9-4be5-bd64-9d3028e65ebd/csds', 'csds/2026/10/08/1230ae44-2525-467d-a9e0-d02975a32deb/csds', 'csds/2026/10/08/12387732-c952-4745-a7e8-952a1d99a6f3/csds', 'csds/2026/10/08/127c28a1-022b-4558-84e7-cf3648a6a577/csds', 'csds/2026/10/08/1dbf4c13-4eea-4d74-ab9a-1496c5521a4c/csds', 'csds/2026/10/08/2823b641-7a00-40c7-b13c-c7e0ecede44f/csds', 'csds/2026/10/08/3efab920-608d-46ec-89ec-ad781ada3f87/csds', 'csds/2026/10/08/51e2e1fc-1641-4a02-bde7-58a041356a4e/csds', 'csds/2026/10/08/5964e9a6-4407-4ff6-abf2-a8a919e7c3cd/csds', 'csds/2026/10/08/5d54909c-6d81-4661-aad3-a68573435187/csds', 'csds/2026/10/08/63007674-4822-436d-ae73-aafe31aaf263/csds', 'csds/2026/10/08/69ae9d35-2810-44f0-9b23-2776b37bd74e/csds', 'csds/2026/10/08/75949240-1d58-4810-a35e-42fcdb60fafc/csds', 'csds/2026/10/08/7ffb7d01-0715-43cb-8973-22275b729a75/csds', 'csds/2026/10/08/812cd6bb-d710-4294-862d-4b98033f3de9/csds', 'csds/2026/10/08/9e90c353-5933-48e8-a5b2-0e84483a36fe/csds', 'csds/2026/10/08/9ee49a0f-7ab0-4a25-80d4-7dfb4cab5a51/csds', 'csds/2026/10/08/a97551d4-4e24-41b7-b882-304038143cab/csds', 'csds/2026/10/08/b52e9964-f5a4-4f8e-9658-0cd26095e292/csds', 'csds/2026/10/08/b803dd68-0c3e-4137-98ef-e6d2b62c522c/csds', 'csds/2026/10/08/c0e044d1-d001-4921-9579-18b8c1214077/csds', 'csds/2026/10/08/cc9b9349-bccc-4851-b37c-2a38461e298c/csds', 'csds/2026/10/08/cfd01ebc-b143-4fdc-addc-f2e6d1b4b787/csds', 'csds/2026/10/08/d685d781-f04b-4565-b89d-445de6ddbbf0/csds', 'csds/2026/10/09/04bf21ae-1443-475c-955a-bdbac605cb93/csds', 'csds/2026/10/09/0b969a58-1eb4-45a1-a5cc-57ff838f4f36/csds', 'csds/2026/10/09/0f16370e-0e1b-465a-becf-95f12ab0442a/csds', 'csds/2026/10/09/1a5bf1ee-70c9-4f69-b834-39b56ed9f139/csds', 'csds/2026/10/09/23005ee7-e8e0-4a4d-b0f8-558d0b6f6cee/csds', 'csds/2026/10/09/42e7198b-d633-4172-b3ed-fbdd2c9f8407/csds', 'csds/2026/10/09/4acc6aab-219e-4ae3-b178-c9a38d4bf29f/csds', 'csds/2026/10/09/4d89dc2a-3e9c-46d2-9c47-5961c6e3be53/csds', 'csds/2026/10/09/5a9de716-bfbb-4592-9417-8d0b77fed6f7/csds', 'csds/2026/10/09/5b073731-c961-442e-b829-925794d1ecc6/csds', 'csds/2026/10/09/65ef56f6-53a0-43a7-8a61-56965e98cc29/csds', 'csds/2026/10/09/71e63c43-a368-467e-91df-88fcf3348235/csds', 'csds/2026/10/09/742603fc-436e-4215-890b-8416eb88b883/csds', 'csds/2026/10/09/878bee9b-5eea-412c-9d44-844c28a3c9bb/csds', 'csds/2026/10/09/91fd3094-4c6c-4c3f-a4ef-648264963ab4/csds', 'csds/2026/10/09/9ac080d7-79b9-441d-ab32-b9951051433f/csds', 'csds/2026/10/09/a55c6218-20d4-47e3-afcf-42ce15c4a0ce/csds', 'csds/2026/10/09/a69ec1e1-7782-4e56-b7b8-74fb5386f8d7/csds', 'csds/2026/10/09/a84630f6-2a6b-462f-a959-45472b88c609/csds', 'csds/2026/10/09/ab6b9454-b804-4ced-93a2-4dec12691827/csds', 'csds/2026/10/09/ad4800a9-8bae-4906-b8b6-07b9c229546e/csds', 'csds/2026/10/09/c04abd6c-11ab-452b-b9a2-91c1ec9ad0c2/csds', 'csds/2026/10/09/c3fdad31-f5a4-481c-b8a9-98b3378be2af/csds', 'csds/2026/10/09/c4acbdf8-63e9-4572-9083-d497f6f13db1/csds', 'csds/2026/10/09/d7850948-c60e-4fcd-ab87-11f28db5fa6a/csds', 'csds/2026/10/09/db256271-c41c-4e6b-a551-cc75328e45dd/csds', 'csds/2026/10/09/ecae8222-e188-4265-b252-5fbc0f9dca4a/csds', 'csds/2026/10/09/f2e34634-383e-4e15-9079-f9beb358dd85/csds', 'csds/2026/10/09/f8e6b141-a0b8-4e18-b496-eed3adcec44c/csds', 'csds/2026/10/10/015f804c-fa99-4682-a80c-7c27e452e454/csds', 'csds/2026/10/10/0918dcd7-78de-4926-94a3-62a96bad17d5/csds', 'csds/2026/10/10/0ea42675-8a37-4f81-9368-1a415e2f2053/csds', 'csds/2026/10/10/17433ca6-2729-4c4e-a9f6-1b85b6a719bf/csds', 'csds/2026/10/10/285cbe82-24cd-4ead-b93f-fab8a1d0817f/csds', 'csds/2026/10/10/33adc667-6994-4057-bf99-c3151a2761f8/csds', 'csds/2026/10/10/37bf28f8-029a-4d69-9242-5f5240cfeb14/csds', 'csds/2026/10/10/3f11d3b4-ee30-43ac-b7e5-4e007ba2df44/csds', 'csds/2026/10/10/4109b71f-7d19-48f6-989c-5327a57f4110/csds', 'csds/2026/10/10/4d50a94b-ce99-465e-8752-2073531ac041/csds', 'csds/2026/10/10/5b7e4c01-1292-43dc-b5be-7ef537c4b796/csds', 'csds/2026/10/10/5dda124a-220d-49d5-8f84-c922ede5e5af/csds', 'csds/2026/10/10/5e1a74ab-01bc-4e2a-998f-8eb62948b607/csds', 'csds/2026/10/10/6037be78-6f30-4eab-8385-73fd785c3427/csds', 'csds/2026/10/10/6ba265e9-685e-4f53-aa7f-2f5df0162243/csds', 'csds/2026/10/10/73a94d0f-6a56-4e42-b34f-8e580492c002/csds', 'csds/2026/10/10/7d9762b2-e355-414d-b054-48465de71cf3/csds', 'csds/2026/10/10/8214bd2f-a1fa-47b9-9966-a5406253fd75/csds', 'csds/2026/10/10/929148c4-cfd8-41e8-819a-839e610f3c8a/csds', 'csds/2026/10/10/9562845d-fc52-479d-8248-289288186456/csds', 'csds/2026/10/10/963a0b30-d332-443e-aa35-3e512bf20fae/csds', 'csds/2026/10/10/b55296f3-5aa0-4e5c-a35d-2e4cee1f39d7/csds', 'csds/2026/10/10/bd7592dc-f413-4671-8a7d-da5f4e95e694/csds', 'csds/2026/10/10/c0f5262b-0764-4c90-927a-1b4490f86b9e/csds', 'csds/2026/10/10/c4b908c3-9d40-420e-a2d5-e84a4e503df5/csds', 'csds/2026/10/10/d50b14f2-ecb6-459d-9803-786a32a8ce96/csds', 'csds/2026/10/10/e10de200-543b-4c32-9e4e-304adcf24d61/csds', 'csds/2026/10/10/ee4538c6-98db-4a3b-a50c-d6180d5d080f/csds', 'csds/2026/10/10/ffe54bb9-102c-4594-ae98-98c22e1c0efe/csds', 'csds/2026/10/11/04137ff8-ac72-42be-b247-0deb4cf76dd1/csds', 'csds/2026/10/11/0519610b-0c0e-466d-9d42-d93b798de6e8/csds', 'csds/2026/10/11/083a39d5-171b-4592-b650-c11ca4955434/csds', 'csds/2026/10/11/0b81377a-2ce6-47e5-b989-6ceccc3a85a4/csds', 'csds/2026/10/11/12dae0c0-b97d-4fcd-83af-1dac048f8590/csds', 'csds/2026/10/11/13895b05-2c5a-449e-8ad9-eb8c4e417bfa/csds', 'csds/2026/10/11/233061d8-d0b8-4f4d-85bd-e9561eb809a0/csds', 'csds/2026/10/11/2e25b3c1-a2bd-4953-abf1-a15714155c46/csds', 'csds/2026/10/11/31b418fb-b921-43a9-85fe-324915e573c3/csds', 'csds/2026/10/11/38025faf-9063-40de-9f4f-4192893b06bb/csds', 'csds/2026/10/11/40c51982-51ec-4e9f-b5db-954ca815addb/csds', 'csds/2026/10/11/46c6ca5e-3d66-4201-9ea4-4e2a718b217c/csds', 'csds/2026/10/11/4929a629-621a-48e5-a550-d32238e02c71/csds', 'csds/2026/10/11/4984cec4-724a-4be6-9860-45db040faf70/csds', 'csds/2026/10/11/57ef6e65-9eaf-43d2-bb3e-b418ed181392/csds', 'csds/2026/10/11/5ff690f7-5adf-4fa4-9da9-4b5834d65dbe/csds', 'csds/2026/10/11/6b45a294-4357-424d-97b3-b8b17a2c0dbb/csds', 'csds/2026/10/11/701261d4-56a3-4053-85d1-679633d175d8/csds', 'csds/2026/10/11/71921323-8588-490d-900d-3ae874aa373a/csds', 'csds/2026/10/11/76f151d1-31ee-4f21-a8d5-811df0e986d2/csds', 'csds/2026/10/11/9e336906-aeb4-4103-ae12-1a54a212faae/csds', 'csds/2026/10/11/a4c4830d-fdc6-4c89-a596-855f21039841/csds', 'csds/2026/10/11/af7bba13-fa7f-471a-845e-ab52236c2065/csds', 'csds/2026/10/11/bc0f3bb7-3ea1-4c46-b769-1f8ce3da4d7b/csds', 'csds/2026/10/11/bca098f5-cf99-4b9b-83ee-565b2ca969c4/csds', 'csds/2026/10/11/c08b5f63-f3b7-41cd-a356-9f6e9ef8ad91/csds', 'csds/2026/10/11/c9eb8a2e-4776-4c55-acec-b0a71af78ac1/csds', 'csds/2026/10/11/dcc08ff8-3e16-4757-a471-6370c50647ef/csds', 'csds/2026/10/11/df5ce63d-5e24-45aa-a4fb-5cf194d88358/csds', 'csds/2026/10/11/e57db62c-9c1a-4261-9928-06e2a5f171c8/csds', 'csds/2026/10/11/fcf059f8-e349-40d9-9cf3-5236f58b71b1/csds', 'csds/2026/10/12/414b387d-645f-4d8d-a05b-72566f86627a/csds', 'csds/2026/10/12/5f0c8917-6d68-40fd-aaf3-f7f3d292607a/csds', 'csds/2026/10/12/86328b45-5c9b-476c-94d4-c697edce312f/csds', 'csds/2026/10/12/926a70b5-026a-48e1-859f-32d077a69b35/csds', 'csds/2026/10/12/a1ce9ab2-5941-4aef-8744-a9b0d8a8f32c/csds', 'csds/2026/10/12/a85c7ff6-e031-45e1-a4ee-96a320d31304/csds', 'csds/2026/10/12/f2f4ff38-9ddb-499e-b815-756d754240f8/csds', 'csds/2026/10/20/4095639f-71a1-4548-9c78-f1b3fa7f3c7c/csds', 'csds/2026/10/20/4a270a5e-ad78-473d-8b62-8036fb88dbdc/csds', 'csds/2026/10/20/4ce6fd68-a8ee-4588-927a-fdfbd5124830/csds', 'csds/2026/10/20/4f42c48a-4ff6-45ed-9e9c-cf9ac8e9ef77/csds', 'csds/2026/10/20/62db2453-69c4-4eca-a88d-390836b8e161/csds', 'csds/2026/10/20/69e4ae9d-9537-4011-a589-359b1565e25e/csds', 'csds/2026/10/20/7fa2018f-4fc2-46a3-9350-efa4b68aad63/csds', 'csds/2026/10/20/867faae2-4314-4fbc-ab7e-fbd832eb989a/csds', 'csds/2026/10/20/8facd8f0-17b6-462e-91b3-75dc1000c6ef/csds', 'csds/2026/10/20/97c1968b-0655-4c11-8756-8deb0d62dd0e/csds', 'csds/2026/10/20/9c1b7693-0fe4-472f-bba5-90c537003998/csds', 'csds/2026/10/20/b3489ee9-b45c-4e42-a1c3-bcc931100625/csds', 'csds/2026/10/20/b6914b6f-2234-4eb5-98a9-c62b811849a0/csds', 'csds/2026/10/20/c9e6e9e7-5858-48a2-8a68-5e0f20de88c9/csds', 'csds/2026/10/20/cd926318-7783-470c-b4fd-f9c5f0efe91f/csds', 'csds/2026/10/20/d4a63283-48a5-4ff5-9649-40831ef6ae55/csds', 'csds/2026/10/20/d73aeb1b-568f-4113-97ee-17a611908d5a/csds', 'csds/2026/10/20/f2cd3e91-c540-4f88-bcd8-ea87fb8d48b9/csds']
arr16 = np.load(OUTD + "loadout_step_i16.npy", mmap_mode="r")
arr_eq = np.load(OUTD + "loadout_equip_i32.npy", mmap_mode="r")
assert arr16.shape == (NS, 2, 19) and arr_eq.shape == (NS, 2)
res = {"hashes": {}}
rounds_by_m = {}
order = np.argsort(rm, kind="stable")
rm_sorted = rm[order]
for mk in SEL_KEYS:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    a = np.ascontiguousarray(np.asarray(arr16[S_all]).astype(np.int16)).tobytes()
    b = np.ascontiguousarray(np.asarray(arr_eq[S_all]).astype(np.int32)).tobytes()
    res["hashes"][mk] = hashlib.sha256(a + b).hexdigest()
log("hashes done", len(res["hashes"]))
# parity rows written by the S3 job vs the arrays as written
pr = pl.read_parquet(STAGE + "parity_s3_rows.parquet")
FIELDS19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
            "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy"]
pk = sorted(pr["match_key"].unique().to_list())
cells = bad = 0
bad_keys = []
for mk in pk:
    m = key_m[mk]
    lo = np.searchsorted(rm_sorted, m, side="left")
    hi = np.searchsorted(rm_sorted, m, side="right")
    rr = np.sort(order[lo:hi])
    S_all = np.concatenate([np.arange(off[r], off[r + 1]) for r in rr])
    sub = pr.filter(pl.col("match_key") == mk)
    pos = np.searchsorted(S_all, sub["s"].to_numpy())
    assert (S_all[pos] == sub["s"].to_numpy()).all()
    dense = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    si = sub["side_idx"].to_numpy()
    dense[pos[:, None], si[:, None], np.arange(20)[None, :]] = np.stack([sub[c].to_numpy() for c in FIELDS19 + ["equip"]], axis=1)
    got = np.zeros((len(S_all), 2, 20), dtype=np.int64)
    got[:, :, :19] = np.asarray(arr16[S_all]).astype(np.int64)
    got[:, :, 19] = np.asarray(arr_eq[S_all]).astype(np.int64)
    d = got != dense
    cells += d.shape[0] * 2
    bad += int(d.any(axis=2).sum())
    if d.any():
        bad_keys.append(mk)
res["parity_rows_vs_written"] = {"keys": len(pk), "side_steps": cells, "mismatch_side_steps": bad, "mismatch_keys": bad_keys[:10]}
log("parity vs written", res["parity_rows_vs_written"])
# ranges and row counts over the whole array
CH = 4_000_000
mx = np.full((2, 19), -10**9, dtype=np.int64)
mn = np.full((2, 19), 10**9, dtype=np.int64)
eq_mx = np.full(2, -10**9, dtype=np.int64)
rows_with_data = np.zeros(2, dtype=np.int64)
steps_with_data = 0
for i in range(0, NS, CH):
    blk = np.asarray(arr16[i:i + CH]).astype(np.int64)
    eb = np.asarray(arr_eq[i:i + CH]).astype(np.int64)
    mx = np.maximum(mx, blk.max(axis=0))
    mn = np.minimum(mn, blk.min(axis=0))
    eq_mx = np.maximum(eq_mx, eb.max(axis=0))
    nz = (blk != 0).any(axis=2) | (eb != 0)
    rows_with_data += nz.sum(axis=0)
    steps_with_data += int(nz.any(axis=1).sum())
res["ranges"] = {"min": mn.tolist(), "max": mx.tolist(), "equip_max": eq_mx.tolist()}
res["rows_with_data_CT_T"] = rows_with_data.tolist()
res["steps_with_any_data"] = steps_with_data
res["total_steps"] = NS
log("ranges and counts", json.dumps(res["ranges"]), res["rows_with_data_CT_T"], steps_with_data)
print("#RESULT " + json.dumps(res), flush=True)
