// Test segmentAt directly
(async function(){
  try {
    var s = window.LM;
    var mod = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5');
    var Tensor = mod.Tensor;
    var points = [[300, 400]];
    var ii = s.imageInputs;
    var oh = ii.original_sizes[0][0], ow = ii.original_sizes[0][1];
    var rh = ii.reshaped_input_sizes[0][0], rw = ii.reshaped_input_sizes[0][1];
    var scaled = points.map(function(p){ return [(p[0]*rw)/ow, (p[1]*rh)/oh]; });
    var input_points = new Tensor('float32', Float32Array.from(scaled.flat()), [1, 1, 2]);
    var input_labels = new Tensor('int64', BigInt64Array.from([1n]), [1, 1]);
    var emb = s.imageEmb;
    var outputs = await s.model(Object.assign({}, ii, emb, {input_points: input_points, input_labels: input_labels}));
    return {
      outKeys: Object.keys(outputs),
      iouScores: outputs.iou_scores ? Array.from(outputs.iou_scores.data) : null,
      predMasksDims: outputs.pred_masks ? outputs.pred_masks.dims : null
    };
  } catch(e) {
    return {err: e.message, stack: e.stack && e.stack.substring(0, 800)};
  }
})()
