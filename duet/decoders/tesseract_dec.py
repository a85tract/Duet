"""Tesseract (Google's A* / beam-search most-likely-error decoder, ``tesseract-decoder``) on the full,
undecomposed circuit detector error model, "long-beam" settings of the Tesseract README / paper."""
from importlib.metadata import version

from tesseract_decoder import tesseract, utils

SETTINGS = {"det_beam": 20, "beam_climbing": True, "no_revisit_dets": True, "pqlimit": 1_000_000,
            "num_det_orders": 21, "det_order": "DetBFS", "det_order_seed": 2384753}


class Tesseract:
    def __init__(self, circuit, **overrides):
        s = {**SETTINGS, **overrides}
        dem = circuit.detector_error_model(decompose_errors=False)
        orders = utils.build_det_orders(dem, s["num_det_orders"], getattr(utils.DetOrder, s["det_order"]),
                                        s["det_order_seed"])
        cfg = tesseract.TesseractConfig(dem=dem, det_beam=s["det_beam"], beam_climbing=s["beam_climbing"],
                                        no_revisit_dets=s["no_revisit_dets"], pqlimit=s["pqlimit"],
                                        det_orders=orders)
        self.dec = cfg.compile_decoder()
        self.settings = {"package": f"tesseract-decoder {version('tesseract-decoder')}", **s}

    def decode(self, dets):
        return self.dec.decode(dets)
