
/*!
\ingroup PkgTriangulationsConcepts
\cgalConcept

This concept describes the geometric types and predicates required to build
a Delaunay triangulation. It corresponds to the first template parameter of the class
`CGAL::Delaunay_triangulation<DelaunayTriangulationTraits_, TriangulationDataStructure_>`.

\cgalRefines{TriangulationTraits}

\cgalHasModelsBegin
\cgalHasModels{CGAL::Epick_d<Dim>}
\cgalHasModels{CGAL::Epeck_d<Dim>}
\cgalHasModelsEnd

\sa `TriangulationTraits`
*/
class DelaunayTriangulationTraits {
public:

/// \name Types
/// @{

/*!
A predicate object that must provide
the templated operator
`template<typename ForwardIterator> Oriented_side operator()(ForwardIterator start, ForwardIterator end, const Point_d & p)`.
The operator returns `ON_POSITIVE_SIDE`,
`ON_NEGATIVE_SIDE`
or `ON_ORIENTED_BOUNDARY` depending of the side of the query
point `p`
with respect to the sphere circumscribing the simplex
defined by the points in range `[start,end)`.
If the simplex is positively
oriented, then the positive side of sphere corresponds geometrically
to its bounded side.
\pre If `Dimension`=`CGAL::Dimension_tag<D>`,
then `std::distance(start,end)=D+1`.
The points in range
`[start,end)` must be affinely independent, i.e., the simplex must
not be flat.
*/
typedef unspecified_type Side_of_oriented_sphere_d;

/*!
A predicate object that must
provide the templated operator
`template<typename ForwardIterator> Oriented_side operator()(Flat_orientation_d orient, ForwardIterator start, ForwardIterator end, const Point_d & p)`.
The operator returns `ON_POSITIVE_SIDE`,
`ON_NEGATIVE_SIDE`
or `ON_ORIENTED_BOUNDARY` depending of the side of the query
point `p`
with respect to the sphere circumscribing the simplex
defined by the points in range `[start,end)`.
If the simplex is positively
oriented according to `orient`,
then the positive side of sphere corresponds geometrically
to its bounded side.
The points in range `[start,end)` and `p` are supposed to belong to the lower dimensional flat
whose orientation is given by `orient`.
\pre `std::distance(start,end)=k+1` where \f$ k\f$ is the number of
points used to construct `orient`.
The points in range
`[start,end)` must be affinely independent, i.e., the simplex must
not be flat. `p` must be in the flat generated by this simplex.
*/
typedef unspecified_type In_flat_side_of_oriented_sphere_d;

/// @}

/// \name Creation
/// @{

/*!
The default constructor (optional).
This is not required when an instance of the traits is provided
to the constructor of `CGAL::Delaunay_triangulation`.
*/
DelaunayTriangulationTraits();

/// @}

/// \name Operations
/// The following methods permit access to the traits class's predicates and functors:
/// @{

/*!

*/
Side_of_oriented_sphere_d side_of_oriented_sphere_d_object() const;

/*!

*/
In_flat_side_of_oriented_sphere_d in_flat_side_of_oriented_sphere_d_object()
const;

/// @}

}; /* end DelaunayTriangulationTraits */
