#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/Variational_shape_reconstruction.h>
#include <iostream>

#include <CGAL/Point_set_3/IO/XYZ.h>
// CGAL
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel     Kernel;


typedef Kernel::FT                  FT;
typedef Kernel::Point_3             Point;
typedef Kernel::Vector_3            Vector;
typedef Kernel::Triangle_3          Triangle;
typedef Kernel::Plane_3             Plane;

typedef Kernel::Point_2             Point_2;
typedef Kernel::Segment_2           Segment_2;

typedef CGAL::First_of_pair_property_map<std::pair<Point, Vector>>                     Point_map;
typedef CGAL::Second_of_pair_property_map<std::pair<Point, Vector>>                    Normal_map;
typedef CGAL::Point_set_3< Point, Vector > Pointset;

int main()
{
    // fixme: this assumes that the normals are read from the file
    Pointset pointset;
    std::string filename;
    std::cin >> filename;
    if (!CGAL::IO::read_XYZ("../../data/" + filename + ".xyz", pointset))
    {
        std::cerr << "Error: cannot read file " << std::endl;
        return EXIT_FAILURE;
    }

    int nb_clusters = 2000;
    int num_knn = 6;

    qem::Variational_shape_reconstruction_ftc vsr(
        pointset,
        nb_clusters,
        num_knn,
        qem::VERBOSE_LEVEL::HIGH);

    //// clustering
    //vsr.clustering();

    // to track clusterings and errors
    std::ofstream file("ftc_errors.csv");
    file << "n_iterations" << "," << "Total Error" << "," << "Variance" << std::endl;
    vsr.clustering_and_compute_errors(file);

    // reconstruction
    const double dist_ratio = 10e-3;
    const double fitting = 0.4;
    const double coverage = 0.3;
    const double complexity = 0.3;
    vsr.reconstruction(dist_ratio, fitting, coverage, complexity, false);

    // save output mesh
    auto mesh = vsr.get_reconstructed_mesh();
    std::ofstream mesh_file;
    mesh_file.open(filename + "_mesh_ftc.off");
    CGAL::write_off(mesh_file, mesh);
    mesh_file.close();

    return EXIT_SUCCESS;
}


