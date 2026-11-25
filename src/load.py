import utils.eegparser as ep


reader = ep.DummyParser("src/config.json")
data = reader.read()
tags = reader.parse_tag_xml("dane/mu_milosz.obci.tag")

print(tags.shape)
print(data.shape)