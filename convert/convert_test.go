package convert

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/llm"
	"golang.org/x/exp/maps"
)

func convertFull(t *testing.T, d string) (*os.File, llm.KV, llm.Tensors) {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), "f16")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := Convert(d, f); err != nil {
		t.Fatal(err)
	}

	r, err := os.Open(f.Name())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })

	m, _, err := llm.DecodeGGML(r)
	if err != nil {
		t.Fatal(err)
	}

	r.Seek(0, io.SeekStart)
	return r, m.KV(), m.Tensors()
}

func TestConvertFull(t *testing.T) {
	cases := []struct {
		path   string
		expect map[string]string
	}{
		{"Meta-Llama-3-8B-Instruct", map[string]string{}},
		{"Mistral-7B-Instruct-v0.2", map[string]string{}},
		{"Mixtral-8x7B-Instruct-v0.1", map[string]string{}},
		{"gemma-2b-it", map[string]string{
			"general.architecture":                   "16a40ab71e67716c31d099d6c540bbf26a01fe17e0102dd12648fac3454d8267",
			"general.file_type":                      "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
			"general.parameter_count":                "29cc276af9689061978af43ec29ff227da3b80fb020ddb1e0515fc8ad8d3bb32",
			"gemma.attention.head_count":             "2c624232cdd221771294dfbb310aca000a0df6ac8b66b696d90ef06fdefb64a3",
			"gemma.attention.head_count_kv":          "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
			"gemma.attention.key_length":             "51e8ea280b44e16934d4d611901f3d3afc41789840acdff81942c2f65009cd52",
			"gemma.attention.layer_norm_rms_epsilon": "159fb29a827ad04b260aa6c8ab6d8637f8f2b38af5c4f3cb49d6a21205e040f8",
			"gemma.attention.value_length":           "51e8ea280b44e16934d4d611901f3d3afc41789840acdff81942c2f65009cd52",
			"gemma.block_count":                      "4ec9599fc203d176a301536c2e091a19bc852759b255bd6818810a42c5fed14a",
			"gemma.context_length":                   "864a936a35324151e1c79c44a2e903ff2497f52fa892282d340585f493c637f0",
			"gemma.embedding_length":                 "bfa0ec8bdf2946547879d50a68687ea32e2fa628db187357415858b633d194d9",
			"gemma.feed_forward_length":              "ca902d4a8acbdea132ada81a004081f51c5c9279d409cee414de5a39a139fab6",
			"tokenizer.ggml.add_bos_token":           "b5bea41b6c623f7c09f1bf24dcae58ebab3c0cdd90ad966bc43a45b44867e12b",
			"tokenizer.ggml.add_eos_token":           "fcbcf165908dd18a9e49f7ff27810176db8e9f63b4352213741664245224f8aa",
			"tokenizer.ggml.bos_token_id":            "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35",
			"tokenizer.ggml.eos_token_id":            "6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
			"tokenizer.ggml.model":                   "1f50ec60f00b2eee3320b6b87ee99ab849359a46fe9d77820af329e53a80b1e9",
			"tokenizer.ggml.padding_token_id":        "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
			"tokenizer.ggml.scores":                  "0872465d173867d755d3ee728f882b9dc2057a0bfd596fe1e3d131522f1250d8",
			"tokenizer.ggml.token_type":              "485e40bf3d715a4764818fc097d6a2a41db872d82ee714bc500872a3437ff48d",
			"tokenizer.ggml.tokens":                  "c6e66de1841f04de8b8d236d461ab720a4c9b9b5414dc293a09c6e10eab45fda",
			"tokenizer.ggml.unknown_token_id":        "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
			"token_embd.weight":                      "33a3f72d4edb3135a9ff5690e042d0659747484c3932ce6841d1fa7430bcd978",
			"blk.0.attn_k.weight":                    "fef221d1b00e28fa7ffbfa49c1bd2235e533f5fa6209d509d53e285d39e51131",
			"blk.0.attn_norm.weight":                 "6ca6e074cee05f59e1f023f507fa4bc0ed6dbd168525cea692fcebddb6013dbe",
			"blk.0.attn_output.weight":               "9e6544275cd7361000299344758a7dc2028871fb9b76f511bc9cf59ed0b7b564",
			"blk.0.attn_q.weight":                    "e12bb596f8fd560714885837860bc1a98878a569b4a7d357cbe449723bd83a6d",
			"blk.0.attn_v.weight":                    "d3c0cf8a911ffc3191170c8918a4f423c7e1942939cd27d95e1992309d35203a",
			"blk.0.ffn_down.weight":                  "7e1476124007783b3512d0fe324123c8acc4129c4e12040be434d8f8a27c6a52",
			"blk.0.ffn_gate.weight":                  "c3a9de98fe51e9e1c0e47e224d4926b377760d83d5c8a8983b4cf554f7328ff4",
			"blk.0.ffn_norm.weight":                  "cbe197e0135a183e31d9646373f1cf99913d6af38a05445f35a5d3cba67abb34",
			"blk.0.ffn_up.weight":                    "fc1a346088601676a97f2dd23efdf436e8d2a65bc208b29d640d24f81a4401f6",
			"blk.1.attn_k.weight":                    "97900ea521529b8553d9659bd54b6737746219509c584e3bbaa5cbd095ab64c8",
			"blk.1.attn_norm.weight":                 "b1dfa432a6a930869c751b96908d2bc674fa5978c834b4e864bc306dc726d48e",
			"blk.1.attn_output.weight":               "3b05b4ddc634170798ad8214a307802a5f0cc5b62240fdcb8269fb16931a43c7",
			"blk.1.attn_q.weight":                    "25d8ea492f426a53c38d78fcbf80efab17926e2e19d026ea83a9169fbc5aaad4",
			"blk.1.attn_v.weight":                    "62e17cf8eb64b6497b4cea4a4e053df651eb42f7d9f4632bddd9f39518d8ddbc",
			"blk.1.ffn_down.weight":                  "a2941656e58db65631f26194f51443862fb6c9a43eaf822ee6e4c82e3076193e",
			"blk.1.ffn_gate.weight":                  "f4a9dca9a74ea2b832fe5368302a964557201c7a4f84abf21a9686b4f884f7eb",
			"blk.1.ffn_norm.weight":                  "68ab8e8d9cfd8c21f7da7bb094e94b7afd86687338c6d8f5bb3275d12d09804a",
			"blk.1.ffn_up.weight":                    "542a82e0762958f9b75a9033db1d14a3019f2669cc10697cdf12fd170514ac90",
			"blk.2.attn_k.weight":                    "ecbb48d660e9813dfca2e7ed7e4156e0a972955149ef829b9a1b443f9457dd16",
			"blk.2.attn_norm.weight":                 "dacd82b8433f01e9e119823fb2068a70fb0f96aae2cb8f10a22359297aa84371",
			"blk.2.attn_output.weight":               "068c3dda7517cf79eb0dc3f2c79ee50a07b663a66e798a307d7fca15f5bb6e4c",
			"blk.2.attn_q.weight":                    "6dacf45e9902878d6b5d57f0f0e66db3e9d2181c17c41747c9c7ca997070fb82",
			"blk.2.attn_v.weight":                    "0858fbee52ebec12a239975bfd27fdb354a60bf75b53852795c22526b9e12444",
			"blk.2.ffn_down.weight":                  "06ccebecd5033590cc1506cd0e38133ffce09b542b10a6f1e61faf17cebe77dc",
			"blk.2.ffn_gate.weight":                  "e60b362c41d30b5c392bcc07d138facca03acde1cc56ff5a9775463da6196f99",
			"blk.2.ffn_norm.weight":                  "03383293c4879be6411bd4eff2b9621d9b29531b4ac9262265b1f303eff69a70",
			"blk.2.ffn_up.weight":                    "f0a6973968baecd50291297f845e1249631c469ea058db9dbd013b81d4273a1e",
			"blk.3.attn_k.weight":                    "a1e3ca75a0ab4debc3a5c34c5beb1d27aaeac46ceaae95fc9ba701fb7c600fc5",
			"blk.3.attn_norm.weight":                 "70622e93b4be30657a5504b0f94daf22aae6f2bda18fe3a5dc1583420d6571de",
			"blk.3.attn_output.weight":               "d99dd24dee4bab29602a6e52dbe99ce5f8a31dedd5c2cfc730586f1a2a616717",
			"blk.3.attn_q.weight":                    "8f6c1205e83e61acf6c51e4680bf0b222f3e04bb2db8cf3170484a920a542aa7",
			"blk.3.attn_v.weight":                    "c6dfb095ca3cac7e77244feba11a241924ccc96bdfc9328dcd0baa60d65f88a7",
			"blk.3.ffn_down.weight":                  "a434cbe58d603cc9d2f472af6988dd20b36b1ee70f4edd8991503dfcc88834a0",
			"blk.3.ffn_gate.weight":                  "6a0cd70c07ee124e04e8a78c267f7874d2fb26825642c373474163841718fe4d",
			"blk.3.ffn_norm.weight":                  "ff9def08f53f9a69b93a88f266aacd537d0ac76fb988aed6eafb3aa97eeb6171",
			"blk.3.ffn_up.weight":                    "1dd83de1ec74d0ee7a7f9465c5a275614a65909528b34fda1d9f9352cbac3ceb",
			"blk.4.attn_k.weight":                    "d8f26cc307adb5d6867a48b00d1142a1be3a6b26086153aef8639146149c82be",
			"blk.4.attn_norm.weight":                 "2c27936a3d34d7c53780c17bbae1dbf0fe4d185370bc1593510531482a9050bf",
			"blk.4.attn_output.weight":               "5545b02205f852efeec388f457dd444457d45fb9b99ac47326521eb2dc8e67ef",
			"blk.4.attn_q.weight":                    "29859a617ffbd2078d601d47b59218c36a3b52afbc0456b32a32398ed58c7667",
			"blk.4.attn_v.weight":                    "b690685897173aba1019ac488f894b9923c329bb14bdb14dbbf9c7dc59572804",
			"blk.4.ffn_down.weight":                  "4c99e90653c8b99d5a9f85ae3d278191ae4f4b7f0e44e7d2e139e03de1befd15",
			"blk.4.ffn_gate.weight":                  "c6d7d6a87ae55c95ba62ecd26651c46e7510cf475c0030c54f8e83db2929b92f",
			"blk.4.ffn_norm.weight":                  "4a4b9fe16d8ed6b569436a39f6abce93122906e7200c2084e9809c8afdfb5cdd",
			"blk.4.ffn_up.weight":                    "d8cd9e117da82800ec7eed637be3941d965da1dddbd90d214d94627781e7fa78",
			"blk.5.attn_k.weight":                    "907bfa9c5a989236b5a514471a2683807d08c04d6f3190940c201b2264cc4f72",
			"blk.5.attn_norm.weight":                 "1d56103885c139f4a8c090849562d898e37668b10da90e6b378ba7c48da04d6c",
			"blk.5.attn_output.weight":               "4a92c0fb9b6d82e4f7d7ec27b16a17996c99d5bfbca8bff8bfc177f99b24b1f2",
			"blk.5.attn_q.weight":                    "e7d5066293a8bce52665630c4c8637d5342863669242ee978aa6f8928ad00e53",
			"blk.5.attn_v.weight":                    "3d9256293d817323978cbeeddef7a70c56f63a72a3bfe7070caf1f06d6395e47",
			"blk.5.ffn_down.weight":                  "62d749562a80760a60120ef0060e67efc2aad8127b8edc09ffe5fdeb00da9876",
			"blk.5.ffn_gate.weight":                  "4f4b6d357adc171b41dbb81e27205644e5832e19fbc143f5c3d31277df77888c",
			"blk.5.ffn_norm.weight":                  "67940d34cf934a12b6c2291f70b6d41337758532c61622ec1c9e03751ff3c50e",
			"blk.5.ffn_up.weight":                    "4bca381c3c631e5776fa9c3dc2fc08e9083a2d37eb1309e04a4a4024848999ea",
			"blk.6.attn_k.weight":                    "ceef3208d968ab55966f8b5ae0700b9f9b9f4979153dced60202218defb61218",
			"blk.6.attn_norm.weight":                 "77fc786dbe6adc6814927a9945985048710a994722a1bb3d308c76ffafd872c1",
			"blk.6.attn_output.weight":               "6e20808b58f40290b804a610901dde65ebfd2c6f698f114342886714c06d1559",
			"blk.6.attn_q.weight":                    "bd703aee7710379d563af2da8b3a166d51a6b2f0b94679a42deb302815a8fccc",
			"blk.6.attn_v.weight":                    "2d1b23d8bf73a8ec8fec171cce394189d1f1224900b556864a9761df2b4d42c7",
			"blk.6.ffn_down.weight":                  "88de7a2f4db10b5f2c90ad0d1c03b7780844c413a62c7de4098accc683f7e3ea",
			"blk.6.ffn_gate.weight":                  "d133a972d3a9d9115bb6da807064ba8b7f6792d5d42ce0cbb348dd36ed074684",
			"blk.6.ffn_norm.weight":                  "b91d858dc122d6fec8c55aedd21e64a69552d37b23cd5d255a82751c3d661f68",
			"blk.6.ffn_up.weight":                    "e54b62d857224d30643e6646b7647c8c99aedc0f432e87cbd136433bc1c7c215",
			"blk.7.attn_k.weight":                    "4570c11b79cb33e1fb309b4d3e94e1113f6ec781d7e9560792df48d4a2c1ec4a",
			"blk.7.attn_norm.weight":                 "1ba8a14aeb7a08ff34e49284aa31400a79dc8b534258c92ee802ea53a83b3b3e",
			"blk.7.attn_output.weight":               "f7a3fff969a2b9bbe9d652350a46609d927804b68c27498c1c01508c08bc3599",
			"blk.7.attn_q.weight":                    "91b305f50acb1b0231cf19f262da7f896d55953a193073636ec071a16f6ffe95",
			"blk.7.attn_v.weight":                    "56d4977423c022e72a2c184ce3709cde75aea38be0d5de5b18a08320b9950cc7",
			"blk.7.ffn_down.weight":                  "5bf43258f7d657c856952a100060f64fb6549e74e4f8b2e6b9c2a2326caf281f",
			"blk.7.ffn_gate.weight":                  "d18a0ee8d1d2415c4761bcd42ae433fd525cbe4a98f534f0eabe00b70a95de22",
			"blk.7.ffn_norm.weight":                  "c11281c55c7774990212c122b1c558be5a8636072ffbd9f7f37cbb2122685be8",
			"blk.7.ffn_up.weight":                    "fb3a02109945861ba9aac46e5dcbd35dc906b6ff8fec9c87de7b77656db89510",
			"blk.8.attn_k.weight":                    "859eb19746419b81af43a7283af37850e2f5d953f2d496db121c94459119cc38",
			"blk.8.attn_norm.weight":                 "9828cef21cf988d16b17e4dbb7ee8ee7c4fa4a190607ec4da5cf250d1647b5c5",
			"blk.8.attn_output.weight":               "ab7deb7bf1be86823afcafdbbf9dbdbdeabce7699a681bc685b46cad636738fe",
			"blk.8.attn_q.weight":                    "c3e5b1ee9bf3645d90073b45c6543c54f609b5960465a309cf2dd716b33dd3c3",
			"blk.8.attn_v.weight":                    "0c35ad2f7d62bdc2d4923d8ae922dae9744b268fd43ffd2dc713ca87a3e07906",
			"blk.8.ffn_down.weight":                  "a4a4648eb7e8c13e10ef1df1bb905da46c179a67dea3a57fc5c6d9d1e3a37fd7",
			"blk.8.ffn_gate.weight":                  "09cdf0bd1310a2ae75d4d1915da25003c6e1c4894f3645c8bdbd297222ec4276",
			"blk.8.ffn_norm.weight":                  "f9ba2c3b44a10e219b108a0fe83f3178aacaea604673c9bb1cb5dfe1892b2404",
			"blk.8.ffn_up.weight":                    "822f539488f6001f860f1979fb838ee4d3745a1ceacebb988e96c26c7493d7a2",
			"blk.9.attn_k.weight":                    "d22302269c2b83adc26033674a035fb22346a327762a151416f8c185e25062f7",
			"blk.9.attn_norm.weight":                 "6360e88a9111849160b2258e3ac24e341641f82c596ee12fbf3593f05a716d29",
			"blk.9.attn_output.weight":               "09294a1fa320af776822ddf2ea007a3b89b2272c1ecd80747fd56f32cf645aa9",
			"blk.9.attn_q.weight":                    "70b444b77d65ac67efd64f75d235b1fa111a518b10ffe1643a5a33396c822e07",
			"blk.9.attn_v.weight":                    "62a7694df280bec758fd0e9dc214d20ff6e57a4ec584d73d97e60bc4165c46bc",
			"blk.9.ffn_down.weight":                  "66e6e67a521a10320454cd6fae3542474e9a0c1d7818f3e2995f80861d4543fb",
			"blk.9.ffn_gate.weight":                  "ff708780fac2a9a775ffad926ec8fea9f9cc75512226b014db4c783a130f2792",
			"blk.9.ffn_norm.weight":                  "239211db69468760a8f2466782923206b75e3c5c472aaccd7236a00b1cc8cf54",
			"blk.9.ffn_up.weight":                    "e160b7826bac28d96e99df5ba7824ebf9a2e90401113928f0b6bb6c7cdda5798",
			"blk.10.attn_k.weight":                   "f7467d7dbe033da5bbb928e4554a30dc7056b204721e0dff1db29303924712e8",
			"blk.10.attn_norm.weight":                "decd25e2cac5f6f3eaaa8c3caf3b096717d4f94802ee35853767f45f96b58f68",
			"blk.10.attn_output.weight":              "b972138143ba7dbe499f46370d31de930442fc7a05750c1dfa464a7f2f2ce5ee",
			"blk.10.attn_q.weight":                   "03fa3c77f1f5a96a7232b67aa83f7c57974e31395497b72c9ef8e4bd4825373e",
			"blk.10.attn_v.weight":                   "ef72c9a08a90e9f143f9d1744ac2a4ac10139d686707db17be2e05ee7c07e4c5",
			"blk.10.ffn_down.weight":                 "ef76f72952fa96b7532198c6b8f6df68e0da148ae38da6dc292927d5a691be85",
			"blk.10.ffn_gate.weight":                 "557f79129bc2410aa9c417d2389557bdad3f8eea5bf1d2b1d183c1ef6d7d7fb4",
			"blk.10.ffn_norm.weight":                 "961401f1eea97be634ac1e5a82b78421beb4b75d5c5bbafc3b812594eaffb84d",
			"blk.10.ffn_up.weight":                   "e6acc01273425d1de3688112e530a291d57633ac149af421238f42f62c7f8d6e",
			"blk.11.attn_k.weight":                   "58ea7f98fad9eeaa99db9e07b0463f2cd0e739047432153d69e171f68b6ea669",
			"blk.11.attn_norm.weight":                "2cb220080edbfa4fed86f8c04f61e2b0fa00137e67454d7e3c9aa28ba4ddc1ee",
			"blk.11.attn_output.weight":              "026492b9a8979ffb7fa822105d4e0835b6dc372c5d2a9c41af060e852a0a2a37",
			"blk.11.attn_q.weight":                   "887f252ca65abf1d8d624cbf39458b26e1339d6fb5151e4b0aeb158ecc1aa164",
			"blk.11.attn_v.weight":                   "32dc5d152cdb985fb23da020294cd1644e89c149c4a959971745be2aa55aae77",
			"blk.11.ffn_down.weight":                 "0036578d002c4712e909a482b4ecad1de672b9998db1262bc9aec562ad7ab8ff",
			"blk.11.ffn_gate.weight":                 "f5f791b9cd5eee20e1d8beed2cb5f8ea2a95a3219a97ee014210d38d2bc00040",
			"blk.11.ffn_norm.weight":                 "205083dc7f71e79b7a5073f7ec0e36c0efbff9fec40f5f35bd8ec617e91c5e19",
			"blk.11.ffn_up.weight":                   "5d8513b17423c7a703504afe693eb326fcc85bdbcdcf92db4299ca9a78f7739e",
			"blk.12.attn_k.weight":                   "50ff35dc77520ff9e55005505f93aaae3768bacd28c53df21d794b23926b839f",
			"blk.12.attn_norm.weight":                "2bbbf421a17fae6e8300ac37e3905fafb3851d739ddcdf7fb29adf10ca56a28c",
			"blk.12.attn_output.weight":              "d52c7d3d8bd60d42379e5f388cbc2cb1bbe7ac5c4dc580921d25863a69082cf2",
			"blk.12.attn_q.weight":                   "e64768017ee888a67661612da742429b8097734e045d9162f20aa4462a83e01e",
			"blk.12.attn_v.weight":                   "dd0202a3bc2d556ff97618865af107a3ff0f57ab82292dac3ef59fb01cfa1725",
			"blk.12.ffn_down.weight":                 "14c35c089ad8afa1a67e87e6436205a3587e47dc9e68e3eabf3b6fde9827370b",
			"blk.12.ffn_gate.weight":                 "1e7611faa62d2debc3f8672c1311838dea66e150f2bda1f7850cd3d8245cb161",
			"blk.12.ffn_norm.weight":                 "fc1ce8140699ec43cad98014dc2bb227e1ec882f4ef0a3f5b8c7958bf2e1eaa7",
			"blk.12.ffn_up.weight":                   "97ce2979c787072f95ef3fce03301f7f533fd9c1bbabd71c8ea21dd4f9b972a1",
			"blk.13.attn_k.weight":                   "da4c52300065a90d83c80e3533c357cf3b8087279ae769f4c8b2a663e4a459bc",
			"blk.13.attn_norm.weight":                "8dd02fe6de0a54f8c05418debd6c47798303de296fc00d5c5c79e33f9b815c5f",
			"blk.13.attn_output.weight":              "92cffc6c89f0f2f395dcc96d1c4037e538b0fb0143801192420c1192288d45d4",
			"blk.13.attn_q.weight":                   "d61707b69bab2d5d271014003701ecfd97486f1da4c20781e2aae5aaf5f2ccf1",
			"blk.13.attn_v.weight":                   "b843bc0922f50a076aa912686db224ea17f2a9ee047e250f4d2d7cd7035de8af",
			"blk.13.ffn_down.weight":                 "bb73f482b2bccd73626a26b3d19bd2f9aba420c74157fab4be9d682bbaee0ec4",
			"blk.13.ffn_gate.weight":                 "8cedbf33519a614c9c44ffa0065d65e05d80f581dfcc7804b951d0ad042da560",
			"blk.13.ffn_norm.weight":                 "d2ff00fe93f42e24adb72be40ee8c8694b9c96da5c17b449731307ef5914bf4e",
			"blk.13.ffn_up.weight":                   "6eee4866f9d880b46f6197391c5e25230a8d4bf4fbb02bdec688cfe0e7eed684",
			"blk.14.attn_k.weight":                   "587bf5ccc492d3445d72182a8d82482cbfed76ddc99f09055fa3ec681516f1cf",
			"blk.14.attn_norm.weight":                "479f3f518229445a1ae74fccf16f2c3817e1287a5fa7c0dc47e55d8b9031d006",
			"blk.14.attn_output.weight":              "dd061f3ab8afe199467f594ee0343f54a0f1034df0d52463f2c21e5f59cb05ce",
			"blk.14.attn_q.weight":                   "7ac1e90c306824b2bd41e029e1f8b2e9068a54cad191fbea18b239ba4db9ef5e",
			"blk.14.attn_v.weight":                   "c5491a482161465fc384fe1c4ec6c211f7e03d968646cad7cf466fb117b66584",
			"blk.14.ffn_down.weight":                 "74af0db248792bb61cd110d58ce79d24edc81e9364e6b8f93c3439b6a308dcd2",
			"blk.14.ffn_gate.weight":                 "c2a3ca49f48700606fa9deec72dad0b0bfd18f67e7f0c6373d5d6120115b409d",
			"blk.14.ffn_norm.weight":                 "38244f55167036d2f422f12d177c464677d13674427b14d8fbd81a95495a957a",
			"blk.14.ffn_up.weight":                   "6f90b71ebb3931e45d60b1a7ceac9ff0605473ae653e2e13e1b4660b444025ad",
			"blk.15.attn_k.weight":                   "17f7d235154d70a30658417ac1b5c5a04cc350036c8a5eed96a057d6205aaa10",
			"blk.15.attn_norm.weight":                "7344adc132e7a30ebca960e7469ab4c22bcc054efe9fda1ffb3897979c983d51",
			"blk.15.attn_output.weight":              "6935365e0e9775c0e7de5b703479e15923c116e978d2cf27d4de69dc8bb58a55",
			"blk.15.attn_q.weight":                   "a6b3e515ecdad2fff6f85b0943671f7f7284adb6529c1e66373dd75d9db4fde2",
			"blk.15.attn_v.weight":                   "6b32489862e847c6106aa1ef0614bf26b2af2234277249e8965840dcdbcc70bf",
			"blk.15.ffn_down.weight":                 "730863d8e309a84ae32218091843435d08c4063cee5b7a586c53fdaaa8e791eb",
			"blk.15.ffn_gate.weight":                 "b012c4e607947d4d2b31398d35716adc1a40d1c79f51ca88214563697087ef07",
			"blk.15.ffn_norm.weight":                 "8a0d89fcecd9f244507a64ee300ad751ede6049b3d1dacf204f6d29bdb8adfac",
			"blk.15.ffn_up.weight":                   "998eef5ee55647f3369af0933072756134265cb3670097d7df01b394b4b8d969",
			"blk.16.attn_k.weight":                   "b326908b0ddc9d2213d8506375de6b8b4a43f81962b9b90fc5fe2de5bed39ff0",
			"blk.16.attn_norm.weight":                "3ca5b5996f8f7b55b0a3e859d360dbc15ef7fd943ced1269fe15c43a3e1b0c15",
			"blk.16.attn_output.weight":              "d701f22c3f272a2c4eea3e7bc79e92286cdd4190063d791da6733374e8f2d451",
			"blk.16.attn_q.weight":                   "15313447c86ce022099f1bc7bba88a538a6f38392e2b5b612b6314785b30de12",
			"blk.16.attn_v.weight":                   "acb8a275bdfa4fb8f6b3bdb98ecae273401903edc16542be4c289cbf017889c3",
			"blk.16.ffn_down.weight":                 "93d53773c034a0d8bf90f2b9631914fbd2c47f37a8423d0be734da415fe6385a",
			"blk.16.ffn_gate.weight":                 "58035709e4321cfc113a817f926ebe2e68b4d4a3e2d639dfeb60e2629d714aea",
			"blk.16.ffn_norm.weight":                 "b9ea7286eeaa6c7736ebdf09faf2406436d57ae20c2129452be1ba940e82919d",
			"blk.16.ffn_up.weight":                   "33e2cf78a441883e9117c266c9212db98dbc3583ac19f02d7a9e6b61af577a76",
			"blk.17.attn_k.weight":                   "991fb2f171010407e81fd9701537ddc8d79eed4ca2b4bed67951eeb1baa428f5",
			"blk.17.attn_norm.weight":                "386208f7ad69e13c8002f922712bdfb82453f4c5da2b95c14e7bba90f2f85277",
			"blk.17.attn_output.weight":              "7a216a824a3584813e1ad56792a2100c25c49c042687a75b4db625ab92869645",
			"blk.17.attn_q.weight":                   "303b40a8fd58e90d9a1cebf912d16ac12e212506586ba313ce5ca20c6842d9d9",
			"blk.17.attn_v.weight":                   "ed61c9f8fd7b045f61a0d3095475bd750927d967331cad074ae4fb4404124aea",
			"blk.17.ffn_down.weight":                 "8a95319b8f6eddd84f58e877cdc722f149b872305e21f34f4baa77d7306c0551",
			"blk.17.ffn_gate.weight":                 "c628cb26243f3e6a8acc847e38a36b59ccd7ab6eefdabd80ed542caac2ee3635",
			"blk.17.ffn_norm.weight":                 "4d4ad17cfda072ff85591200badd47b8ed7311ea94037f5b46d2aaa596bb44a5",
			"blk.17.ffn_up.weight":                   "7ed8588c1d13472467f8fd35b2fdbf65a35b5fcd361a0fe3f3f3a1e2f5f406df",
			"output_norm.weight":                     "582db6c175ea9657f4b29b2304d8d281655218adc878b9dafc7dc91a915a90bb",
		}},
	}

	for _, tt := range cases {
		t.Run(tt.path, func(t *testing.T) {
			p := filepath.Join("testdata", tt.path)
			if _, err := os.Stat(p); err != nil {
				t.Skipf("%s not found", p)
			}

			f, kv, tensors := convertFull(t, p)
			actual := make(map[string]string)
			for k, v := range kv {
				bts, err := json.Marshal(v)
				if err != nil {
					t.Fatal(err)
				}

				actual[k] = fmt.Sprintf("%x", sha256.Sum256(bts))
			}

			for _, tensor := range tensors {
				sha256sum := sha256.New()
				sr := io.NewSectionReader(f, int64(tensor.Offset), int64(tensor.Size()))
				if _, err := io.Copy(sha256sum, sr); err != nil {
					t.Fatal(err)
				}

				actual[tensor.Name] = fmt.Sprintf("%x", sha256sum.Sum(nil))
			}

			keys := maps.Keys(tt.expect)
			slices.Sort(keys)
			for _, k := range keys {
				if tt.expect[k] != actual[k] {
					t.Errorf("unexpected %s: want %s, got %s", k, tt.expect[k][:12], actual[k][:12])
				}
			}
		})
	}
}
