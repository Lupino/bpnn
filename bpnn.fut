-- bpnnet run on GPU
-- use https://futhark-lang.org
-- compile:
--    futhark opencl bpnn.fut
--    futhark c bpnn.fut

def sigmoid (x: f32) = f32.tanh x
def dsigmoid (y: f32) = 1.0 - y**2

type Unit [n] = {
  weight:    [n]f32,
  change:    [n]f32,
  sample:    [n]f32,
  threshold: f32
}

def new_unit [n] (weight: [n]f32) (threshold: f32) = {
  weight = weight,
  change = replicate n (0.0:f32),
  sample = replicate n (0.0:f32),
  threshold = threshold
}

def unit_update_sample [n] (unit: Unit[n]) (sample: [n]f32) =
  unit with sample = sample

def unit_calc [n] (unit: Unit[n]) =
  sigmoid (f32.sum (map2 (*) unit.sample unit.weight) - unit.threshold)

def unit_update [n] (unit: Unit[n]) (diff: f32) (rate: f32) (factor: f32) =
  let change = (map2 (\x c -> rate * x * diff + factor * c) unit.sample unit.change)
  let weight =  map2 (+) unit.weight change
  in unit with weight = weight
          with change = map (diff *) unit.sample

--         input output
type Layer [n] [m] = {
  units: [m]Unit[n]
}

def new_layer [n] [m] (units: [m]Unit[n]) = {units = units}

def layer_update_sample [n] [m] (layer: Layer[n][m]) (sample: [n]f32) =
  layer with units = map (\unit -> unit_update_sample unit sample) layer.units

def layer_calc [n] [m] (layer: Layer[n][m]) = map unit_calc layer.units

def layer_update [n] [m] (layer: Layer[n][m]) (diffs: [m]f32) (rate: f32) (factor: f32) =
  layer with units = map2 (\x unit -> unit_update unit x rate factor) diffs layer.units

def layer_get_error_inner [n] [m] (layer: Layer[n][m]) (deltas: [m]f32) (i: i64) =
  f32.sum (map2 (\delta unit -> delta * unit.weight[i]) deltas layer.units)

def layer_get_error [n] [m] (layer: Layer[n][m]) (deltas: [m]f32) =
  map (\i -> layer_get_error_inner layer deltas i) (0..<n)

type BPNNet [ni] [nh] [no] = {
  hlayer: Layer[ni][nh],
  olayer: Layer[nh][no],
  ai: [ni]f32,
  ah: [nh]f32,
  ao: [no]f32
}

def new_bpnnet [ni] [nh] [no] (hlayer: Layer[ni][nh]) (olayer: Layer[nh][no]) = {
  hlayer = hlayer,
  olayer = olayer,
  ai = replicate ni (0.0:f32),
  ah = replicate nh (0.0:f32),
  ao = replicate no (0.0:f32)
}

def bpnnet_calc [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (sample: [ni]f32) =
  let hlayer = layer_update_sample bpnnet.hlayer sample
  let ah = layer_calc hlayer
  let olayer = layer_update_sample bpnnet.olayer ah
  let ao = layer_calc olayer
  in bpnnet with hlayer = hlayer
            with olayer = olayer
            with ai = sample
            with ah = ah
            with ao = ao

def bpnnet_update [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (targets: [no]f32) (rate: f32) (factor: f32) =
  let output_deltas = map2 (\target ao -> dsigmoid(ao) * (target - ao)) targets bpnnet.ao
  let hiddin_deltas = map2 (\ah error -> dsigmoid(ah) * error) bpnnet.ah (layer_get_error bpnnet.olayer output_deltas)
  let olayer = layer_update bpnnet.olayer output_deltas rate factor
  let hlayer = layer_update bpnnet.hlayer hiddin_deltas rate factor
  in bpnnet with hlayer = hlayer
            with olayer = olayer

def bpnnet_get_error [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (targets: [no]f32) =
  f32.sum (map2 (\t o -> 0.5 * (t - o) ** 2) targets bpnnet.ao)

type Pattern [n] [m] = {
  input: [n]f32,
  target: [m]f32
}

def bpnnet_train_iter_one [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (pattern: Pattern[ni][no]) (rate: f32) (factor: f32) =
  let net = bpnnet_calc bpnnet pattern.input
  let onet = bpnnet_update net pattern.target rate factor
  let error = bpnnet_get_error net pattern.target
  in (onet, error)

def bpnnet_train_iter [np] [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (patterns: [np]Pattern[ni][no]) (rate: f32) (factor: f32) =
  foldl (\(net, error) pattern ->
    let (onet, oerror) = bpnnet_train_iter_one net pattern rate factor in (onet, oerror + error))
    (bpnnet, 0.0:f32) patterns

def bpnnet_train [np] [ni] [nh] [no] (bpnnet: BPNNet[ni][nh][no]) (patterns: [np]Pattern[ni][no]) (iters: i32) (rate: f32) (factor: f32) =
  foldl (\(net, error) _ ->
    let (onet, oerror) = bpnnet_train_iter net patterns rate factor in (onet, oerror + error))
    (bpnnet, 0.0:f32) (0..<iters)

def test_net =
  -- let unit0 = new_unit [0.01, 0.02] 0.01
  let unit1 = new_unit [0.03, 0.07] 0.03
  -- let unit2 = new_unit [0.13, 0.03] 0.07
  let unit3 = new_unit [-0.03, 0.03, -0.07] 0.05
  let unit4 = new_unit [-0.01, -0.03, 0.05] (-0.03:f32)
  let layer0 = new_layer [unit3, unit4]
  let layer1 = new_layer [unit1]
  in new_bpnnet layer0 layer1


def test_patterns =
  let pattern0 = {input = [0.0,0.0,1.0:f32], target=[0.0:f32]}
  let pattern1 = {input = [0.0,1.0,1.0:f32], target=[1.0:f32]}
  let pattern2 = {input = [1.0,0.0,1.0:f32], target=[1.0:f32]}
  let pattern3 = {input = [1.0,1.0,1.0:f32], target=[0.0:f32]}
  in [pattern0, pattern1, pattern2, pattern3]

def main (iters: i32) =
  let err = (bpnnet_train test_net test_patterns iters 0.4 0.1).1
  in err / f32.i32 iters
