import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "build"))

import saltro_py


# ============================================================================
# GeometryFace Tests
# ============================================================================

def test_geometry_face_default_constructor():
    """Test default constructor initializes all values to zero."""
    face = saltro_py.GeometryFace()
    
    assert face.area == 0.0
    assert np.allclose(face.centroid, np.zeros(3))
    assert np.allclose(face.normal, np.zeros(3))
    assert face.eta_s == 0.0
    assert face.eta_d == 0.0
    assert face.eta_a == 0.0
    assert face.CD == 0.0


def test_geometry_face_constructor_with_required_parameters():
    """Test constructor with only required parameters."""
    area = 0.1
    centroid = np.array([0.5, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    
    face = saltro_py.GeometryFace(area, centroid, normal)
    
    assert face.area == area
    assert np.allclose(face.centroid, centroid)
    assert np.allclose(face.normal, normal)
    assert face.eta_s == 0.0
    assert face.eta_d == 0.0
    assert face.eta_a == 0.0
    assert face.CD == 0.0


def test_geometry_face_constructor_with_all_parameters():
    """Test constructor with all parameters."""
    area = 0.1
    centroid = np.array([0.5, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    eta_s = 0.1
    eta_d = 0.5
    eta_a = 0.4
    CD = 2.2
    
    face = saltro_py.GeometryFace(area, centroid, normal, eta_s, eta_d, eta_a, CD)
    
    assert face.area == area
    assert np.allclose(face.centroid, centroid)
    assert np.allclose(face.normal, normal)
    assert face.eta_s == eta_s
    assert face.eta_d == eta_d
    assert face.eta_a == eta_a
    assert face.CD == CD


def test_geometry_face_attributes_are_mutable():
    """Test that GeometryFace attributes can be modified."""
    face = saltro_py.GeometryFace()
    
    # Modify all attributes
    face.area = 0.5
    face.centroid = np.array([1.0, 2.0, 3.0])
    face.normal = np.array([0.0, 1.0, 0.0])
    face.eta_s = 0.2
    face.eta_d = 0.3
    face.eta_a = 0.5
    face.CD = 2.5
    
    # Verify changes
    assert face.area == 0.5
    assert np.allclose(face.centroid, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(face.normal, np.array([0.0, 1.0, 0.0]))
    assert face.eta_s == 0.2
    assert face.eta_d == 0.3
    assert face.eta_a == 0.5
    assert face.CD == 2.5


# ============================================================================
# GeometryConfig Tests
# ============================================================================

def test_geometry_config_default_constructor():
    """Test default constructor creates empty configuration."""
    config = saltro_py.GeometryConfig()
    
    assert config.numFaces == 0
    assert len(config) == 0


def test_geometry_config_add_single_face():
    """Test adding a single face."""
    config = saltro_py.GeometryConfig()
    face = saltro_py.GeometryFace(0.1, np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    
    result = config.addFace(face)
    
    assert result is True
    assert config.numFaces == 1
    assert len(config) == 1


def test_geometry_config_add_multiple_faces():
    """Test adding multiple faces."""
    config = saltro_py.GeometryConfig()
    
    for i in range(5):
        normal = np.zeros(3)
        normal[i % 3] = 1.0
        face = saltro_py.GeometryFace(0.1 * (i + 1), np.zeros(3), normal)
        result = config.addFace(face)
        assert result is True
    
    assert config.numFaces == 5
    assert len(config) == 5


def test_geometry_config_add_face_up_to_maximum():
    """Test adding faces up to maximum capacity."""
    config = saltro_py.GeometryConfig()
    max_faces = saltro_py.GeometryConfig.maxFaces()
    
    # Should be able to add max_faces
    for i in range(max_faces):
        face = saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        result = config.addFace(face)
        assert result is True
    
    assert config.numFaces == max_faces


def test_geometry_config_add_face_beyond_maximum():
    """Test that adding face beyond maximum returns False."""
    config = saltro_py.GeometryConfig()
    max_faces = saltro_py.GeometryConfig.maxFaces()
    
    # Fill to capacity
    for i in range(max_faces):
        face = saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        config.addFace(face)
    
    # Try to add one more - should return False
    extra_face = saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    result = config.addFace(extra_face)
    
    assert result is False
    assert config.numFaces == max_faces


def test_geometry_config_get_face_with_valid_index():
    """Test getting a face with valid index."""
    config = saltro_py.GeometryConfig()
    
    area = 0.5
    centroid = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 1.0])
    face = saltro_py.GeometryFace(area, centroid, normal, 0.1, 0.2, 0.3, 2.2)
    
    config.addFace(face)
    
    retrieved_face = config.getFace(0)
    
    assert retrieved_face.area == area
    assert np.allclose(retrieved_face.centroid, centroid)
    assert np.allclose(retrieved_face.normal, normal)
    assert retrieved_face.eta_s == 0.1
    assert retrieved_face.eta_d == 0.2
    assert retrieved_face.eta_a == 0.3
    assert retrieved_face.CD == 2.2


def test_geometry_config_get_face_with_invalid_index():
    """Test that getting face with invalid index raises exception."""
    config = saltro_py.GeometryConfig()
    config.addFace(saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0])))
    
    # Index too large
    with pytest.raises(Exception):
        config.getFace(1)
    
    with pytest.raises(Exception):
        config.getFace(10)


def test_geometry_config_get_face_when_empty():
    """Test that getting face from empty config raises exception."""
    config = saltro_py.GeometryConfig()
    
    with pytest.raises(Exception):
        config.getFace(0)


def test_geometry_config_clear():
    """Test clearing all faces."""
    config = saltro_py.GeometryConfig()
    
    # Add some faces
    for i in range(5):
        face = saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        config.addFace(face)
    
    assert config.numFaces == 5
    
    # Clear
    config.clear()
    
    assert config.numFaces == 0
    assert len(config) == 0


def test_geometry_config_clear_allows_adding_again():
    """Test that after clearing, faces can be added again."""
    config = saltro_py.GeometryConfig()
    
    # Add and clear
    config.addFace(saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0])))
    config.clear()
    
    # Add again
    face = saltro_py.GeometryFace(0.2, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    result = config.addFace(face)
    
    assert result is True
    assert config.numFaces == 1


def test_geometry_config_max_faces_constant():
    """Test that maxFaces returns the expected constant."""
    max_faces = saltro_py.GeometryConfig.maxFaces()
    
    # Should be MAX_NUM_GEOMETRY_FACES from limits
    assert max_faces == 20  # As defined in limits.h
    assert max_faces > 0


def test_geometry_config_indexing_operator():
    """Test accessing faces using indexing operator []."""
    config = saltro_py.GeometryConfig()
    
    # Add faces with different areas
    for i in range(3):
        face = saltro_py.GeometryFace(0.1 * (i + 1), np.zeros(3), np.array([1.0, 0.0, 0.0]))
        config.addFace(face)
    
    # Access using indexing
    assert np.isclose(config[0].area, 0.1)
    assert np.isclose(config[1].area, 0.2)
    assert np.isclose(config[2].area, 0.3)


def test_geometry_config_indexing_out_of_range():
    """Test that indexing out of range raises IndexError."""
    config = saltro_py.GeometryConfig()
    config.addFace(saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0])))
    
    with pytest.raises(IndexError):
        _ = config[1]


def test_geometry_config_iteration():
    """Test iterating over faces."""
    config = saltro_py.GeometryConfig()
    
    # Add faces with incremental areas
    areas = [0.1, 0.2, 0.3, 0.4, 0.5]
    for area in areas:
        face = saltro_py.GeometryFace(area, np.zeros(3), np.array([1.0, 0.0, 0.0]))
        config.addFace(face)
    
    # Iterate and collect areas
    collected_areas = [face.area for face in config]
    
    assert collected_areas == areas


def test_geometry_config_len_operator():
    """Test len() operator on GeometryConfig."""
    config = saltro_py.GeometryConfig()
    
    assert len(config) == 0
    
    config.addFace(saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0])))
    assert len(config) == 1
    
    config.addFace(saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0])))
    assert len(config) == 2


def test_geometry_config_modify_face_through_reference():
    """Test modifying a face through getFace reference."""
    config = saltro_py.GeometryConfig()
    
    face = saltro_py.GeometryFace(0.1, np.zeros(3), np.array([1.0, 0.0, 0.0]))
    config.addFace(face)
    
    # Get reference and modify
    retrieved_face = config.getFace(0)
    retrieved_face.area = 0.5
    retrieved_face.eta_s = 0.8
    
    # Verify changes persist
    assert config.getFace(0).area == 0.5
    assert config.getFace(0).eta_s == 0.8


def test_geometry_config_different_face_configurations():
    """Test adding faces with different configurations."""
    config = saltro_py.GeometryConfig()
    
    # Face 1: Front face
    face1 = saltro_py.GeometryFace(
        area=0.04,  # 20cm x 20cm
        centroid=np.array([0.1, 0.0, 0.0]),
        normal=np.array([1.0, 0.0, 0.0]),
        eta_s=0.1,
        eta_d=0.5,
        eta_a=0.4,
        CD=2.2
    )
    
    # Face 2: Solar panel
    face2 = saltro_py.GeometryFace(
        area=0.2,
        centroid=np.array([0.0, 0.5, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        eta_s=0.8,  # Highly reflective
        eta_d=0.1,
        eta_a=0.1,
        CD=1.5
    )
    
    config.addFace(face1)
    config.addFace(face2)
    
    assert config.numFaces == 2
    assert config[0].area == 0.04
    assert config[1].area == 0.2
    assert config[0].eta_s == 0.1
    assert config[1].eta_s == 0.8
