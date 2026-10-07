"""Exercise the adapter's configuration without requiring a Slicer runtime."""
import ast
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import bone_contouring

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'HRpQCTTools/SegmentationHRpQCT/SegmentationHRpQCT.py'


def adapter_namespace():
    tree = ast.parse(SOURCE.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                {'SITE_PRESETS', 'METHOD_PRESETS'} for t in node.targets):
            nodes.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == '_contour_site_defaults':
            nodes.append(node)
        if isinstance(node, ast.ClassDef):
            nodes.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and
                         n.name in {'_generate_bone_masks_with_bone_contouring', '_collect_params',
                                    '_apply_params_to_widgets', '_on_contour_method_changed',
                                    '_effective_contour_site'})
    ns = {'asdict': asdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
def test_slicer_defaults_match_core_standard(site, modality):
    ns = adapter_namespace()
    default = ns['_contour_site_defaults'](site, modality)
    core = bone_contouring.resolve_preset(modality=modality, site=site, segmentation='gauss')
    assert default['outer']['periosteal_threshold'] == core.outer.periosteal_threshold
    assert default['inner']['endosteal_threshold'] == core.inner.endosteal_threshold
    assert default['outer']['gaussian_sigma'] == core.outer.gaussian_sigma
    assert default['inner']['gaussian_sigma'] == core.inner.gaussian_sigma
    assert default['inner']['trabecular_close_radius'] == core.inner.trabecular_close_radius


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('site', ['radius', 'tibia', 'knee'])
def test_standard_peel_is_six_for_all_sites(modality, site):
    defaults = adapter_namespace()['_contour_site_defaults'](site, modality)
    assert defaults['inner']['peel'] == 6


@pytest.mark.parametrize('peel', [None, 0, 5])
@pytest.mark.parametrize('modality,site,outer_threshold,inner_threshold,kernel', [
    ('xct2', 'radius', 320, 500, (31, 31, 1)),
    ('xct1', 'radius', 250, 500, (10, 10, 1)),
    ('xct1', 'tibia', 250, 500, (10, 10, 1)),
    ('xct1', 'knee', 150, 150, (10, 10, 1)),
    ('xct2', 'knee', 150, 150, (10, 10, 1)),
])
def test_adapter_passes_new_kernel_and_physical_smoothing_and_keeps_overrides(
        monkeypatch, modality, site, outer_threshold, inner_threshold, kernel, peel):
    captured = []
    class StopAtGeneration(Exception):
        pass
    def capture(image, params, **kwargs):
        captured.append(params)
        raise StopAtGeneration
    monkeypatch.setattr(bone_contouring, 'generate_masks_from_image', capture)
    logic = SimpleNamespace(_import_bone_contouring=lambda: bone_contouring)
    call = adapter_namespace()['_generate_bone_masks_with_bone_contouring']
    with pytest.raises(StopAtGeneration):
        call(logic, None, None, site=site, segmentation_method='seg_gauss',
             periosteal_contour_method='standard', endosteal_contour_method='standard',
             params={'modality': modality, 'segmentation': {'cort_threshold': 470, 'keep_largest_component': True},
                     'inner': {} if peel is None else {'peel': peel},
                     'stable_3d': {'inner_sigma_mm': [0.03, 0.03, 0.04]}})
    p = captured[0]
    assert p.outer.periosteal_threshold == outer_threshold
    assert p.inner.endosteal_threshold == inner_threshold
    default_peel = 6
    assert p.inner.peel == (default_peel if peel is None else peel)
    assert p.buie.endosteal_kernel_size == kernel
    assert tuple(p.stable_3d.inner_sigma_mm) == (.03, .03, .04)
    assert p.stable_3d.outer_sigma_mm == (.03, .03, .06)
    assert p.segmentation.cort_threshold == 470
    assert p.segmentation.gaussian_sigma == .8
    assert p.segmentation.gaussian_support == 1
    assert p.segmentation.trab_threshold == 320
    assert not p.segmentation.keep_largest_component


@pytest.mark.parametrize('outer,inner', [('standard', 'standard'), ('none', 'none')])
def test_recipe_roundtrip_preserves_physical_settings_and_supports_no_contours(outer, inner):
    ns = adapter_namespace()
    class Widget:
        logic = SimpleNamespace(_import_bone_contouring=lambda: bone_contouring)
        _scannerPreset = 'xct2'
        siteCombo = SimpleNamespace(currentData='radius')
        periostealContourCombo = SimpleNamespace(currentData=outer)
        endostealContourCombo = SimpleNamespace(currentData=inner)
        def _effective_modality(self):
            return 'xct2'
        def __getattr__(self, key):
            return SimpleNamespace(value=1, checked=True, currentData='cpu')
    widget = Widget()
    ns['_apply_params_to_widgets'](widget, {'stable_3d': {'inner_sigma_mm': [.03, .03, .04]},
                                           'segmentation': {'gaussian_support': 2}})
    exported = ns['_collect_params'](widget)
    assert exported['stable_3d']['inner_sigma_mm'] == [.03, .03, .04]
    assert exported['segmentation']['gaussian_support'] == 2
    if inner == 'standard':
        assert exported['buie']['endosteal_kernel_size'] == (31, 31, 1)


@pytest.mark.parametrize('modality', ['xct1', 'xct2'])
@pytest.mark.parametrize('detected', ['radius', 'tibia', 'knee', 'unparsed'])
@pytest.mark.parametrize('method', ['none', 'standard'])
def test_auto_recipe_collection_and_effective_site(detected, method, modality):
    ns = adapter_namespace()
    class Widget:
        _scannerPreset = modality
        siteCombo = SimpleNamespace(currentData='auto')
        periostealContourCombo = SimpleNamespace(currentData=method)
        endostealContourCombo = SimpleNamespace(currentData=method)
        logic = SimpleNamespace(_import_bone_contouring=lambda: bone_contouring)
        volumeSelector = SimpleNamespace(currentNode=lambda: None)
        _contourRegularizationParams = {}
        def _effective_modality(self):
            return modality
        def _selected_site(self, **kwargs):
            return detected
        def _effective_contour_site(self):
            return ns['_effective_contour_site'](self)
        def __getattr__(self, key):
            return SimpleNamespace(value=1, checked=True, currentData='cpu')
    widget = Widget()
    assert widget._effective_contour_site() == detected
    exported = ns['_collect_params'](widget, site='auto')
    assert exported['modality'] == modality
    assert ('buie' in exported) == (method == 'standard' and detected != 'unparsed')
    if 'stable_3d' in exported:
        assert exported['stable_3d']['inner_sigma_mm'] == (.03, .03, .06)


@pytest.mark.parametrize('modality,site,threshold,sigma', [
    ('xct2', 'radius', 320, .8), ('xct2', 'tibia', 320, .8),
    ('xct1', 'radius', 250, 1.5), ('xct1', 'tibia', 250, 1.5),
    ('xct1', 'knee', 150, 1.5), ('xct2', 'knee', 150, 1.5),
])
def test_switching_to_standard_initializes_only_the_changed_stage(modality, site, threshold, sigma):
    ns = adapter_namespace()
    outer_threshold = SimpleNamespace(value=999.)
    inner_threshold = SimpleNamespace(value=500.)
    widget = SimpleNamespace(
        _lastContourMethods=('none', 'standard'),
        _suppressMethodCustomSwitch=False,
        _scannerPreset=modality,
        _effective_modality=lambda: modality,
        periostealContourCombo=SimpleNamespace(currentData='standard'),
        endostealContourCombo=SimpleNamespace(currentData='standard'),
        _effective_contour_site=lambda: site,
        _mark_contour_methods_custom=lambda: None, _refresh_method_dependent_ui=lambda: None,
        periostealThresholdSpin=outer_threshold, endostealThresholdSpin=inner_threshold,
        outerGaussSigmaSpin=SimpleNamespace(value=1.5), innerGaussSigmaSpin=SimpleNamespace(value=1.5))
    ns['_on_contour_method_changed'](widget)
    assert outer_threshold.value == threshold
    assert widget.outerGaussSigmaSpin.value == sigma
    assert inner_threshold.value == 500.  # Existing expert override is preserved.
