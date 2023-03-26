// Copyright (c) 2007-2020  INRIA (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Jackson Campolattaro, Cédric Portaneri, Tong Zhao

#ifndef CGAL_ORTHTREE_NODE_H
#define CGAL_ORTHTREE_NODE_H

#include <CGAL/license/Orthtree.h>

#include <CGAL/Orthtree.h>

#include <boost/range/iterator_range.hpp>

#include <array>
#include <memory>
#include <bitset>
#include <cassert>
#include <iostream>

namespace CGAL {

/*!

  \brief represents a single node of the tree. Alternatively referred
  to as a cell, orthant, or sub-tree.

  A `Node` is a lightweight object and thus generally passed by
  copy. It is also a model of `ConstRange` with value type `Traits::Point_d`.

  \cgalModels `ConstRange`
 */
template <typename Traits, typename PointRange, typename PointMap>
class Orthtree<Traits, PointRange, PointMap>::Node {

public:

  /// \name Types
  /// @{

  typedef Orthtree<Traits, PointRange, PointMap> Enclosing; ///< Orthtree type (enclosing class).
  typedef typename Enclosing::Dimension Dimension; ///< Dimension type.
  typedef typename Enclosing::Degree Degree; ///< Degree type.

  /*!
    \brief Self typedef for convenience.
   */
  typedef typename Orthtree<Traits, PointRange, PointMap>::Node Self;


  /// \cond SKIP_IN_MANUAL
  /*!
   * \brief Array for containing the child nodes of this node.
   */
  typedef std::array<Self, Degree::value> Children;
  /// \endcond

  /*!
    \brief Set of bits representing this node's relationship to its parent.

    Equivalent to an array of Booleans, where index[0] is whether `x`
    is greater, index[1] is whether `y` is greater, index[2] is whether
    `z` is greater, and so on for higher dimensions if needed.
    Used to represent a node's relationship to the center of its parent.
   */
  typedef std::bitset<Dimension::value> Local_coordinates;

  /*!
    \brief Coordinates representing this node's relationship
    with the rest of the tree.

    Each value `(x, y, z, ...)` of global coordinates is calculated by doubling
    the parent's global coordinates and adding the local coordinates.
   */
  typedef std::array<std::uint32_t, Dimension::value> Global_coordinates;


  typedef typename PointRange::const_iterator const_iterator; ///< constant iterator type.

  /*!
    \brief Adjacency directions.
   */
  typedef typename Traits::Adjacency Adjacency;

  /// @}

private:

  /// \cond SKIP_IN_MANUAL
  /*!
    \brief point range type.
   */
  typedef typename PointRange::iterator iterator; ///< constant iterator type.
  typedef boost::iterator_range<iterator> Point_range;
  /// \endcond

  Point_range m_points;
  Self* m_parent = nullptr; // todo: use optional<reference_wrapper<Self>> instead of Self *
  std::uint8_t m_depth = 0;
  Global_coordinates m_global_coordinates{};
  std::shared_ptr<Children> m_children{};


  // Only the Orthtree class has access to the non-default
  // constructor, mutators, etc.
  friend Enclosing;

public: // todo: Was there a good reason that all of this was private?

  /*!
   * \brief Access to the content held by this node
   * \return a reference to the collection of point indices
   */
  Point_range& points() { return m_points; }

  const Point_range& points() const { return m_points; }

  /// \name Construction
  /// @{

  /// \cond SKIP_IN_MANUAL
  Node() = default;
  /// \endcond

  /*!
    \brief creates a new node, optionally as the child of a parent

    If no parent is provided, the node created is assumed to be the
    root of a tree.  This means that `parent.is_null()` returns
    `true`, and the depth is zero.  If a parent is provided, the node
    becomes the child of that parent.  In that case, an index should
    be passed, telling this node its relationship to its parent.
    Depth and global coordinates are automatically determined in the
    constructor, and should generally be considered immutable after
    construction.

    \param parent the node containing this one
    \param index this node's relationship to its parent
  */
  explicit Node(Self* parent, Local_coordinates local_coordinates)
    : m_parent(parent) {

//    if (m_parent) {
//      m_depth = m_parent->m_depth + 1;
//
//      for (int i = 0; i < Dimension::value; i++)
//        m_global_coordinates[i] = (2 * parent->m_global_coordinates[i]) + local_coordinates[i];
//    }

    if (parent != nullptr) {
      m_depth = parent->m_depth + 1;

      for (int i = 0; i < Dimension::value; i++)
        m_global_coordinates[i] = (2 * parent->m_global_coordinates[i]) + local_coordinates[i];

    } else {
      m_depth = 0;

      for (int i = 0; i < Dimension::value; i++)
        m_global_coordinates[i] = 0;
    }
  }

  /// @}

public:

  /// \name Type & Location
  /// @{

  /*!
    \brief returns `true` if the node is null, `false` otherwise.
  */
  //bool is_null() const { return (m_data == nullptr); }

  /*!
    \brief returns `true` if the node has no parent, `false` otherwise.
    \pre `!is_null()`
  */
  bool is_root() const {
    return m_parent == nullptr;
  }

  /*!
    \brief returns `true` if the node has no children, `false` otherwise.
    \pre `!is_null()`
  */
  bool is_leaf() const {
    return (!m_children);
  }

  /*!
    \brief returns this node's depth.
    \pre `!is_null()`
  */
  std::uint8_t depth() const {
    return m_depth;
  }

  /*!
    \brief returns this node's local coordinates (in relation to its parent).
    \pre `!is_null()`
  */
  Local_coordinates local_coordinates() const {

    Local_coordinates result;

    for (std::size_t i = 0; i < Dimension::value; ++i)
      result[i] = global_coordinates()[i] & 1;

    return result;
  }

  /*!
    \brief returns this node's global coordinates.
    \pre `!is_null()`
  */
  Global_coordinates global_coordinates() const {
    return m_global_coordinates;
  }


  /// \name Adjacency
  /// @{

  /*!
    \brief returns this node's parent.
    \pre `!is_root()`
   */
  const Self* parent() const {
    CGAL_precondition (!is_root());
    return m_parent;
  }

  /*!
    \brief returns the nth child of this node.

    \pre `!is_leaf()`
    \pre `0 <= index && index < Degree::value`

    The orthtree subdivides the space in 2 on each dimension
    available, so a child node can be accessed by selecting a Boolean
    value for each dimension. The `index` parameter is thus
    interpreted as a bitmap, where each bit matches a dimension
    (starting by the least significant bit for coordinate X).

    For example, in the case of an octree (dimension 3):

    - index 0 (000 in binary) is the children on the "minimum corner" (xmin, ymin, zmin)
    - index 1 (001 in binary) is on (xmax, ymin, zmin)
    - index 2 (010 in binary) is on (xmin, ymax, zmin)
    - index 6 (101 in binary) is on (xmax, ymin, zmax)

    Diagram of a quadtree:

    ```
    Children of the current node:

    Y axis
    A
    |  +-------------------+-------------------+
    |  | Coord:  Ymax Xmin | Coord:  Ymax Xmax |
    |  | Bitmap:    1    0 | Bitmap:    1    1 |
    |  |                   |                   |
    |  | -> index = 2      | -> index = 3      |
    |  |                   |                   |
    |  |                   |                   |
    |  |                   |                   |
    |  |                   |                   |
    |  +-------------------+-------------------+
    |  | Coord:  Ymin Xmin | Coord:  Ymin Xmax |
    |  | Bitmap:    0    0 | Bitmap:    0    1 |
    |  |                   |                   |
    |  | -> index = 0      | -> index = 1      |
    |  |                   |                   |
    |  |                   |                   |
    |  |                   |                   |
    |  |                   |                   |
    |  +-------------------+-------------------+
    |
    +--------------------------------------------> X axis
    0
    ```

    The operator can be chained. For example, `n[5][2][3]` returns the
    third child of the second child of the fifth child of a node `n`.
  */
  Self& operator[](std::size_t index) {

    CGAL_precondition (!is_leaf());
    CGAL_precondition (index < Degree::value);

    return (*m_children)[index];
  }

  /*!
    \brief returns the nth child of this node.

    \pre `!is_leaf()`
    \pre `index < Degree::value`

    The operator can be chained. For example, `n[5][2][3]` returns the
    third child of the second child of the fifth child of a node `n`.
   */
  const Self& operator[](std::size_t index) const {

    CGAL_precondition (!is_leaf());
    CGAL_precondition (index < Degree::value);

    return (*m_children)[index];
  }

  /*!
    \brief finds the directly adjacent node in a specific direction

    \pre `!is_null()`
    \pre `direction.to_ulong < 2 * Dimension::value`

    Adjacent nodes are found according to several properties:
    - adjacent nodes may be larger than the seek node, but never smaller
    - a node has at most `2 * Dimension::value` different adjacent nodes (in 3D: left, right, up, down, front, back)
    - adjacent nodes are not required to be leaf nodes

    Here's a diagram demonstrating the concept for a Quadtree:

    ```
    +---------------+---------------+
    |               |               |
    |               |               |
    |               |               |
    |       A       |               |
    |               |               |
    |               |               |
    |               |               |
    +-------+-------+---+---+-------+
    |       |       |   |   |       |
    |   A   |  (S)  +---A---+       |
    |       |       |   |   |       |
    +---+---+-------+---+---+-------+
    |   |   |       |       |       |
    +---+---+   A   |       |       |
    |   |   |       |       |       |
    +---+---+-------+-------+-------+
    ```

    - (S) : Seek node
    - A  : Adjacent node

    Note how the top adjacent node is larger than the seek node.  The
    right adjacent node is the same size, even though it contains
    further subdivisions.

    This implementation returns the adjacent node if it's found.  If
    there is no adjacent node in that direction, it returns a null
    node.

    \param direction which way to find the adjacent node relative to
    this one. Each successive bit selects the direction for the
    corresponding dimension: for an Octree in 3D, 010 means: negative
    direction in X, position direction in Y, negative direction in Z.

    \return the adjacent node if it exists, a null node otherwise.
  */
  const Self* adjacent_node(Local_coordinates direction) const {

    // Direction:   LEFT  RIGHT  DOWN    UP  BACK FRONT
    // direction:    000    001   010   011   100   101

    // Nodes only have up to 2*dim different adjacent nodes (since cubes have 6 sides)
    CGAL_precondition(direction.to_ulong() < Dimension::value * 2);

    // The root node has no adjacent nodes!
    if (is_root()) return nullptr;

    // The least significant bit indicates the sign (which side of the node)
    bool sign = direction[0];

    // The first two bits indicate the dimension/axis (x, y, z)
    uint8_t dimension = uint8_t((direction >> 1).to_ulong());

    // Create an offset so that the bit-significance lines up with the dimension (e.g. 1, 2, 4 --> 001, 010, 100)
    int8_t offset = (uint8_t) 1 << dimension;

    // Finally, apply the sign to the offset
    offset = (sign ? offset : -offset);

    // Check if this child has the opposite sign along the direction's axis
    if (local_coordinates()[dimension] != sign) {
      // This means the adjacent node is a direct sibling, the offset can be applied easily!
      return &(*parent())[local_coordinates().to_ulong() + offset];
    }

    // Find the parent's neighbor in that direction, if it exists
    const Self* adjacent_node_of_parent = parent()->adjacent_node(direction);

    // If the parent has no neighbor, then this node doesn't have one
    if (!adjacent_node_of_parent) return nullptr;

    // If the parent's adjacent node has no children, then it's this node's adjacent node
    if (adjacent_node_of_parent->is_leaf())
      return adjacent_node_of_parent;

    // Return the nearest node of the parent by subtracting the offset instead of adding
    return &(*adjacent_node_of_parent)[local_coordinates().to_ulong() - offset];

  }

  /*!
    \brief equivalent to `adjacent_node()`, with an adjacency direction
    rather than a bitset.
   */
  const Self* adjacent_node(Adjacency adjacency) const {
    return adjacent_node(std::bitset<Dimension::value>(static_cast<int>(adjacency)));
  }

  /*!
   * \brief equivalent to adjacent_node, except non-const
   */
  Self* adjacent_node(std::bitset<Dimension::value> direction) {
    return const_cast<Self*>(const_cast<const Self*>(this)->adjacent_node(direction));
  }

  /*!
   * \brief equivalent to adjacent_node, with a Direction rather than a bitset and non-const
   */
  Self* adjacent_node(Adjacency adjacency) {
    return adjacent_node(std::bitset<Dimension::value>(static_cast<int>(adjacency)));
  }

  /// @}

  /// \name Point Range
  /// @{

  /*!
    \brief checks whether the node is empty of points or not.
   */
  bool empty() const {
    return m_points.empty();
  }

  /*!
    \brief returns the number of points of this node.
   */
  std::size_t size() const {
    return std::size_t(std::distance(m_points.begin(), m_points.end()));
  }

  /*!
    \brief returns the iterator at the start of the collection of
    points held by this node.
   */
  const_iterator begin() const { return m_points.begin(); }

  /*!
    \brief returns the iterator at the end of the collection of
    points held by this node.
   */
  const_iterator end() const { return m_points.end(); }

  /// @}

  /// \cond SKIP_IN_MANUAL
  /// \name Operators
  /// @{

  /*!
   * \brief compares the topology of this node to another node.
   *
   * \todo This seems out of date, the implementation I see compares for direct equality
   *
   * \param rhs node to compare with
   * \return whether the nodes have different topology.
   */
  bool operator==(const Self& rhs) const {

    // todo: This is a trivial implementation, maybe it can be set to =default in c++17?
    return rhs.m_parent == m_parent &&
           rhs.m_children == m_children &&
           rhs.m_points == m_points &&
           rhs.m_depth == m_depth &&
           rhs.m_global_coordinates == m_global_coordinates;
  }

  // todo: this does what the documentation for operator== claims to do!
  static bool is_topology_equal(const Self& a, const Self& b) {

    // If one node is a leaf, and the other isn't, they're not the same
    if (a.is_leaf() != b.is_leaf())
      return false;

    // If both nodes are non-leaf nodes
    if (!a.is_leaf()) {

      // Check all the children
      for (int i = 0; i < Degree::value; ++i) {
        // If any child cell is different, they're not the same
        if (!is_topology_equal(a[i], b[i]))
          return false;
      }
    }

    // If both nodes are leaf nodes, they must be in the same location
    return (a.global_coordinates() == b.global_coordinates());
  }

  friend std::ostream& operator<<(std::ostream& os, const Self& node) {
    return internal::print_orthtree_node(os, node);
  }
  /// \endcond
};

}

#endif //CGAL_ORTHTREE_NODE_H
