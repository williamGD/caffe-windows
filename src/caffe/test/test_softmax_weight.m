caffe.reset_all();
caffe.set_mode_gpu();
net = caffe.Net('test_softmax_weight.prototxt','train');
feature = single(randn(2,100));
class_label = single(randn(1,100) > randn(1,100));
weights = single([zeros(1,10) ones(1,90)]);%single(randn(1,100) > randn(1,100));%single(ones(1,100));
f = net.forward({feature; class_label; weights});
g = net.backward({1});