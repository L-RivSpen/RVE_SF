
// Required Packages                                      //
#include <deal.II/base/function.h>                        //
#include <deal.II/base/parameter_handler.h>               // needed for parameters (shocking)
#include <deal.II/base/patterns.h>                        // needed for parameters 
#include <deal.II/base/quadrature_lib.h>                  // Used for quadrature points
#include <deal.II/base/quadrature_point_data.h>           // used for storing cell data
#include <deal.II/base/timer.h>                           // Timer
#include <deal.II/base/work_stream.h>                     // needed for parrallel processing
                                                          //
                                                          //
// Dof Tools                                              // 
#include <deal.II/dofs/dof_renumbering.h>                 // DOFs
#include <deal.II/dofs/dof_tools.h>                       // DOFs
                                                          //
                                                          //
// Meshing Tools                                          //
#include <deal.II/grid/grid_generator.h>                  // for triangulation
#include <deal.II/grid/grid_tools.h>                      // for triangulation
#include <deal.II/grid/grid_in.h>                         // for triangulation
#include <deal.II/grid/grid_out.h>                        //
#include <deal.II/grid/tria.h>                            // for triangulation
                                                          //
                                                          //
// Mapping and Quadratures                                //
#include <deal.II/fe/component_mask.h>                    // 
#include <deal.II/fe/fe_dgp.h>                            //
#include <deal.II/fe/fe_values_extractors.h>              //
#include <deal.II/fe/fe_q.h>                              //
#include <deal.II/fe/fe_system.h>                         //
#include <deal.II/fe/fe_tools.h>                          //
#include <deal.II/fe/fe_values.h>                         //
#include <deal.II/fe/mapping_q_eulerian.h>                //
                                                          //
                                                          //
// Matrix Math                                            //
#include <deal.II/lac/block_sparse_matrix.h>              // needed for block vectors
#include <deal.II/lac/block_vector.h>                     // needed for block vectors
#include <deal.II/lac/dynamic_sparsity_pattern.h>         //
#include <deal.II/lac/linear_operator_tools.h>            //
#include <deal.II/lac/matrix_out.h>                       // Visualizing matices
#include <deal.II/lac/precondition.h>                     //
#include <deal.II/lac/precondition_selector.h>            // needed for preconditioners
#include <deal.II/lac/solver_cg.h>                        // Solver Conjugate gradient
#include <deal.II/lac/solver_gmres.h>                     //
#include <deal.II/lac/solver_minres.h>                    // Uses Minres
#include <deal.II/lac/solver_control.h>                   // needed for solver controls
#include <deal.II/lac/solver_selector.h>                  // Solver selector
#include <deal.II/lac/sparse_direct.h>                    //
#include <deal.II/lac/vector_memory.h>                    // needed for growing vector memory (For solvers)
#include <deal.II/lac/la_parallel_block_vector.h>

                                                          //
                                                          //
// Vector and number tools                                // 
#include <deal.II/numerics/data_out.h>                    // needed for outputting data 
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>                // used for "project" vector
#include <deal.II/numerics/data_out_dof_data.templates.h> // used for output
                                                          //
                                                          //
// Physics                                                //  
#include <deal.II/physics/elasticity/standard_tensors.h>  // needed for material modeling
#include <deal.II/physics/elasticity/kinematics.h>        // needed for material modeling
                                                          //
                                                          //
// Automatic Differentiation                              //
#include <deal.II/differentiation/ad.h>                   // needed for automatic differentiation
                                                          //
                                                          //
// Basics for C++                                         //
#include <iostream>                                       //
#include <fstream>                                        //


namespace RVE
{
using namespace dealii;

// Parameters

namespace Parameters
{

// Finite Element System Parameters
struct FESystem
{
    unsigned int poly_degree;
    unsigned int quad_order;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void FESystem::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        prm.declare_entry("Polynomial degree",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");
        
        prm.declare_entry("Quadrature order",
                          "3",
                           Patterns::Integer(0),
                           "Gauss quadrature order");
    }
    prm.leave_subsection();
}

void FESystem::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order  = prm.get_integer("Quadrature order");
    }
    prm.leave_subsection();
}

// Geometry Parameters
struct Geometry
{
    unsigned int global_refinement;
    double       scale;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Geometry::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry");
    {
        prm.declare_entry("Global refinement",
                          "2",
                          Patterns::Integer(0),
                          "Global refinement level");

        prm.declare_entry("Grid scale",
                          "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");
    }
    prm.leave_subsection();
}

void Geometry::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Geometry");
    {
        global_refinement = prm.get_integer("Global refinement");
        scale             = prm.get_double("Grid scale");
    }
    prm.leave_subsection();
}

// Material Parameters
struct Materials
{
    double nu;
    double mu;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Materials::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        prm.declare_entry("Poisson's ratio",
                          "0.4999",
                          Patterns::Double(-1.0,0.5),
                          "Poisson's ratio");
        prm.declare_entry("Shear modulus",
                          "80.194e6",
                          Patterns::Double(),
                          "Shear modulus");
    }
    prm.leave_subsection();
}

void Materials::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        nu = prm.get_double("Poisson's ratio");
        mu = prm.get_double("Shear modulus");
    }
    prm.leave_subsection();
}

// Linear Solver Parameters
struct LinearSolver
{
    std::string type_lin;
    double      tol_lin;
    double      max_iterations_lin;
    std::string preconditioner_type;
    double      preconditioner_relaxation;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void LinearSolver::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Linear solver");
    {
        prm.declare_entry("Solver type",
                          "CG",
                          Patterns::Selection("CG|Direct|GMRES|Other"),
                          "Type of solver used to solve the linear system");
        
        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry("Max iteration multiplier",
                          "1",
                          Patterns::Double(0.0),
                          "Linear solver iterations (multiples of the system matrix size)");
        
        prm.declare_entry("Preconditioner type",
                          "ssor",
                          Patterns::Selection("jacobi|ssor"),
                          "Type of preconditioner");
        
        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");
    }
    prm.leave_subsection();
}

void LinearSolver::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Linear solver");
    {
        type_lin = prm.get("Solver type");
        tol_lin = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
        preconditioner_type = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
    }
    prm.leave_subsection();
}

// Non-Linear Solver
struct NonlinearSolver
{
    unsigned int max_iterations_NR;
    double       tol_f;
    double       tol_u;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void NonlinearSolver::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Nonlinear solver");
    {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
    }
    prm.leave_subsection();
}

void NonlinearSolver::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Nonlinear solver");
    {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
    }
    prm.leave_subsection();
}

// Psuedo Time

struct Time
{
    double delta_t;
    double end_time;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Time::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time");
    {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time step size", "0.1", Patterns::Double(), "Time step size");
    }
    prm.leave_subsection();
}

void Time::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time");
    {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
    }
    prm.leave_subsection();
}

// Code Parameters
struct Code
{
    bool terminal_output_mode;
    bool file_output_mode;

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);
};

void Code::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Code");
    {
        prm.declare_entry("Terminal Output",
                          "true",
                          Patterns::Bool(),
                          "Enables or disables debugging outputs in terminal");

        prm.declare_entry("File Output",
                          "false",
                          Patterns::Bool(),
                          "Enables or disables files for debugging outputs");
    }
    prm.leave_subsection();
}

void Code::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Code");
    {
        terminal_output_mode = prm.get_bool("Terminal Output");
        file_output_mode     = prm.get_bool("File Output");
    }
    prm.leave_subsection();
}


// All Parameters

struct AllParameters : public FESystem, 
                       public Geometry, 
                       public Materials, 
                       public LinearSolver, 
                       public NonlinearSolver,
                       public Time,
                       public Code

{

    ParameterHandler prm;

    AllParameters(const std::string &input_file);

    static void declare_parameters(ParameterHandler &prm);

    void parse_parameters(ParameterHandler &prm);

    void print_parameters();
};

AllParameters::AllParameters(const std::string &input_file)
{
    declare_parameters(prm);
    prm.parse_input(input_file);
    parse_parameters(prm);
    print_parameters();
}

void AllParameters::declare_parameters(ParameterHandler &prm)
{
    FESystem::declare_parameters(prm);
    Geometry::declare_parameters(prm);
    Materials::declare_parameters(prm);
    LinearSolver::declare_parameters(prm);
    NonlinearSolver::declare_parameters(prm);
    Time::declare_parameters(prm);
    Code::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm)
{
    FESystem::parse_parameters(prm);
    Geometry::parse_parameters(prm);
    Materials::parse_parameters(prm);
    LinearSolver::parse_parameters(prm);
    NonlinearSolver::parse_parameters(prm);    
    Time::parse_parameters(prm);
    Code::parse_parameters(prm);

}

// Parameter Console Output 

void AllParameters::print_parameters()
{
    
    std::cout << '\n' << "--------------------------------------------------------------" << std::endl;
    // FESystem
    prm.enter_subsection("Finite element system");
    {
        std::cout << "Polynomial Degree = " << poly_degree << '\n'
        << "Quadrature Order = " << quad_order << std::endl;
    }
    prm.leave_subsection();

    // Geometry
    prm.enter_subsection("Geometry");
    {
        std::cout << "Global Refinement = " << global_refinement << '\n'
        << "Grid Scale = " << scale << std::endl;
    }
    prm.leave_subsection();

    // Materials
    prm.enter_subsection("Material properties");
    {
        std::cout << "Poisson's Ratio = " << nu << '\n'
        << "Shear Modulus = " << mu << std::endl;
    }
    prm.leave_subsection();

    // Linear solver
    prm.enter_subsection("Linear solver");
    {
        std::cout << "Solver Type = " << type_lin << '\n'
        << "Residual = " << tol_lin << '\n' 
        << "Max Iteration Multiplier = " << max_iterations_lin << '\n'  
        << "Preconditioner Type = " << preconditioner_type << '\n'
        << "Preconditioner Relaxation = " << preconditioner_relaxation << std::endl; 
    }
    prm.leave_subsection();

    // Nonlinear Solver
    prm.enter_subsection("Nonlinear solver");
    {
        std::cout << "Max Iterations Newton-Raphson = " << max_iterations_NR << '\n'
        << "Tolerance Force = " << tol_f << '\n'
        << "Tolerance Displacement = " << tol_u << std::endl;        
    }
    prm.leave_subsection();

    // Psuedo Time
    prm.enter_subsection("Time");
    {
        std::cout << "End Time = " << end_time << '\n'
        << "Time Step Size = " << delta_t << std::endl;     
    }
    prm.leave_subsection();

    // Psuedo Code
    prm.enter_subsection("Code");
    {
        std::cout << "Terminal Output = " << terminal_output_mode << '\n'
        << "File Output = " << file_output_mode << std::endl;     
    }
    prm.leave_subsection();

    std::cout << "--------------------------------------------------------------" << '\n' << std::endl;
}
} // namespace Parameters


// Material Model
template <int dim>
class Material
{
public:
    Material(const double mu, const double nu)
        : kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)))
        , c_1(mu / 2.0)
    {
        Assert(kappa > 0, ExcInternalError());
    }

    void update_material(Tensor<2, dim> &new_F)
    {
        F = new_F;
        compute_derivatives_ad();
    }

    Tensor<2, dim> compute_P()
    {
        return P;
    }

    Tensor<2, dim> analytical_P()
    {
        const double J     = determinant(F);
        const Tensor<2,dim>  b  = F * transpose(F);
        const double         I1 = trace(b);
        const Tensor<2,dim> F_invT = transpose(invert(F));
        
        // Isochoric part
        Tensor<2,dim> P_iso = 2.0 * c_1 * F;

        // Volumetric part
        const double p_vol = 0.5 * kappa * (J*J - 1.0);
        Tensor<2,dim> P_vol = p_vol * F_invT;

        return P_iso + P_vol;
    }

    Tensor<4, dim> compute_tangent()
    {
        return mat_tangent;
    }

    Tensor<4, dim> analytical_mat()
    {
        // Compute J = det(F) and F^{-T}.
        const double       J      = determinant(F);
        const Tensor<2, dim> F_inv = invert(F);
        Tensor<2, dim>     F_invT;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
            F_invT[i][j] = F_inv[j][i];

        // Precompute J^2 and (J^2 - 1).
        const double J2      = J * J;
        const double J2_minus_1 = J2 - 1.0;

        Tensor<4, dim> A; // Automatically initialized to zero.

        // Loop over all indices: (i,I) for P and (j,J) for F.
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int I = 0; I < dim; ++I)
            for (unsigned int j = 0; j < dim; ++j)
                for (unsigned int J_index = 0; J_index < dim; ++J_index)
                {
                // 1) Deviatoric / “2C1 delta” term:
                //      2*C1 * delta_{i j} * delta_{I J_index} 
                const double dev_term
                    = ( (i == j) && (I == J_index) )
                    ? 2.0 * c_1
                    : 0.0;

                // 2) Volumetric “K J^2 (F^{-T})_{iI} (F^{-T})_{jJ}” term:
                const double vol_termA
                    = kappa * J2 
                    * F_invT[i][I] 
                    * F_invT[j][J_index];

                // 3) Volumetric “- (K/2)(J^2 - 1) (F^{-T})_{iJ} (F^{-T})_{jI}” term:
                const double vol_termB
                    = -0.5 * kappa 
                    * J2_minus_1 
                    * F_invT[i][J_index] 
                    * F_invT[j][I];

                A[i][I][j][J_index] = dev_term + vol_termA + vol_termB;
                }

        return A;

    }

private:
    // SEF Function
    template <typename NumberType>
    NumberType compute_SEF(const Tensor<2, dim, NumberType> &F)
    {
        NumberType J_tilde = determinant(F);
        Tensor<2, dim, NumberType> F_bar = Physics::Elasticity::Kinematics::F_iso(F);
        const Tensor<2, dim, NumberType> b_bar_tensor = F_bar * transpose(F_bar);
        NumberType tr_b = trace(b_bar_tensor);

        // Volumetric Response
        NumberType G = 0.25 * (J_tilde * J_tilde - 1 - 2 * std::log(J_tilde));
        NumberType SEF_vol = kappa * G;

        // Isochoric Response
        NumberType SEF_iso = c_1 * (tr_b - dim);

        // Total Response
        return SEF_iso + SEF_vol;
    }

    // AD function to compute SEF and derivatives
    void compute_derivatives_ad()
    {
        // Setup AD helper
        constexpr unsigned int n_independent_variables = dim * dim;
        constexpr Differentiation::AD::NumberTypes ADTypeCode =
            Differentiation::AD::NumberTypes::sacado_dfad_dfad;
        using ADHelper = Differentiation::AD::ScalarFunction<dim, ADTypeCode, double>;
        using ADNumberType = typename ADHelper::ad_type;

        ADHelper ad_helper(n_independent_variables);

        // Register independent variables (flatten F)
        std::vector<double> F_flat(n_independent_variables);
        unsigned int idx = 0;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                F_flat[idx++] = F[i][j];

        ad_helper.register_independent_variables(F_flat);

        // Rebuild F_ad
        Tensor<2, dim, ADNumberType> F_ad;
        const auto &sens = ad_helper.get_sensitive_variables();
        idx = 0;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                F_ad[i][j] = sens[idx++];

        // Compute SEF and register as dependent variable
        const ADNumberType psi_ad = compute_SEF(F_ad);
        ad_helper.register_dependent_variable(psi_ad);

        // Compute derivatives
        Vector<double> dPsi_dF(n_independent_variables);
        FullMatrix<double> ddPsi_dF2(n_independent_variables, n_independent_variables);

        ad_helper.compute_gradient(dPsi_dF);
        ad_helper.compute_hessian(ddPsi_dF2);

        // Rebuild P (first Piola-Kirchhoff stress)
        idx = 0;
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
                P[i][j] = dPsi_dF[idx++];

        // Rebuild tangent modulus
        for (unsigned int i = 0; i < dim; ++i)
            for (unsigned int j = 0; j < dim; ++j)
            {
                unsigned int row = i * dim + j;
                for (unsigned int k = 0; k < dim; ++k)
                    for (unsigned int l = 0; l < dim; ++l)
                    {
                        unsigned int col = k * dim + l;
                        mat_tangent[i][j][k][l] = ddPsi_dF2(row, col);
                    }
            }
    }

    // Deformation
    Tensor<2, dim> F;

    // Constitutive model parameters
    const double kappa;
    const double c_1;

    // First Piola Kirchhoff Stress
    Tensor<2, dim> P;

    // Tangent Tensor
    Tensor<4, dim> mat_tangent;
};



// Time 
class Time
{
public:
  Time(const double time_end, const double delta_t)
    : timestep(0)
    , time_current(0.0)
    , time_end(time_end)
    , delta_t(delta_t)
  {}

  virtual ~Time() = default;

  double current() const
  {
    return time_current;
  }
  double end() const
  {
    return time_end;
  }
  double get_delta_t() const
  {
    return delta_t;
  }
  unsigned int get_timestep() const
  {
    return timestep;
  }
  void increment()
  {
    time_current += delta_t;
    ++timestep;
  }

private:
  unsigned int timestep;
  double       time_current;
  const double time_end;
  const double delta_t;
};

template <int dim>
class MacroDisplacement : public Function<dim>
{
  public:
    MacroDisplacement(const Tensor<2,dim> &F)
      : Function<dim>(dim), F(F) {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &values) const override
    {
      Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
      Tensor<1, dim> up = F * p;        // **use p itself**, not a constant “dir”
      for (unsigned int d = 0; d < dim; ++d)
        values[d] = up[d];
    }

  private:
    Tensor<2, dim> F;
};



// Problem class
template <int dim>
class RVE_SF
{
    public:

        RVE_SF(const std::string &input_file);
        void run();

    private:

        // Functions
        void make_grid();
        void setup_system();
        void assemble_system();
        void apply_boundary_conditions();
        void solve_linear_system();
        void solve_nonlinear();
        void output_results();
        void move_mesh();
        Tensor<2, dim> compute_avg_deformation();
        Tensor<2, dim> compute_avg_stress();
        

        // Core Data Members
        Parameters::AllParameters parameters;
        Triangulation<dim> triangulation;
        FESystem<dim>          fe;
        DoFHandler<dim>    dof_handler;
        QGauss<dim> quadrature;
        AffineConstraints<double> constraints;
        SparsityPattern sparsity_pattern;
        SparseMatrix<double> tangent_matrix;
        Vector<double> system_rhs;
        Vector<double> solution;

        // Time 
        Time time;

        // Material 
        Material<dim> material;

        // Boundary Values
        // List for storing pair BC's
        std::vector<unsigned int> left_dofs, right_dofs, bottom_dofs, 
                                  top_dofs, back_dofs, front_dofs;


        // Fixed Dofs
        unsigned int fixed_dof1_x, fixed_dof1_y, fixed_dof2_x,
                     fixed_dof1_z, fixed_dof2_y, fixed_dof3_x;

        std::vector<unsigned int> fixed_dofs;

        // Resulting Strain
        std::map<types::global_dof_index, double> boundary_values;
        Tensor<2, dim> Fbar = unit_symmetric_tensor<dim>();

        // Total deformation for mapping geometry
        Vector<double> u_total;
       


        // Output values
        Tensor<2, dim> avg_deformation;
        Tensor<2, dim> avg_stress;
};

template <int dim>
RVE_SF<dim>::RVE_SF(const std::string &input_file)
    : parameters(input_file)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , quadrature(parameters.quad_order)
    , time(parameters.end_time, parameters.delta_t)
    , material(parameters.mu, parameters.nu)
    {
        Assert(dim == 2 || dim ==3, ExcMessage("This problem only works in 2 or 3 dimensions"));
        u_total.reinit(dof_handler.n_dofs());
    }



template <int dim>
void RVE_SF<dim>::run()
{
    make_grid();
    setup_system();
    
    while(time.current() < time.end())
    {
        apply_boundary_conditions();
        solve_nonlinear();


        avg_deformation = compute_avg_deformation();
        avg_stress = compute_avg_stress();

        std::cout << "Average Deformation: \n   " << avg_deformation << std::endl;
        std::cout << "Average Stress: \n   " << avg_stress << std::endl;
        
        move_mesh();

        output_results();
        time.increment();
    };
    


};

template <int dim>
void RVE_SF<dim>::make_grid()
{
    // Mesh Generation
    GridGenerator::hyper_cube(triangulation, 0, 1, true);
    GridTools::scale(parameters.scale, triangulation);
    triangulation.refine_global(parameters.global_refinement);

    std::cout << "Mesh Generated" << std::endl;
    
    
}

template <int dim>
void RVE_SF<dim>::setup_system()
{
    std::cout << "Setting Up System:" << std::endl;
    std::cout << " - Distributing DOFs" << std::endl;
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);

    std::cout << " - Generating Sparsity Pattern" << std::endl;
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_dofs=*/true);
    sparsity_pattern.copy_from(dsp);

    std::cout << " - Resizing Tangent Matrix and RHS and Solution Vectors" << std::endl;
    tangent_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    u_total.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    solution = 0;

    std::cout << " - Finding Perioidic Faces" << std::endl;

    
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);


    const double tol = 1e-12; // Tolerance for floating point comparison
    for (unsigned int i = 0; i < support_points.size(); ++i)
    {
        const auto &p = support_points[i];
        if (std::abs(p[0] - 0.0) < tol)
            left_dofs.push_back(i);
        if (std::abs(p[0] - parameters.scale) < tol)
            right_dofs.push_back(i);
        if (std::abs(p[1] - 0.0) < tol)
            bottom_dofs.push_back(i);
        if (std::abs(p[1] - parameters.scale) < tol)
            top_dofs.push_back(i);
        if (dim == 3)
            {
                if (std::abs(p[2] - 0.0) < tol)
                    back_dofs.push_back(i);
                if (std::abs(p[2] - parameters.scale) < tol)
                    front_dofs.push_back(i);
            }
    }

    std::cout << " - Finding DOFs to fix" << std::endl;

    std::map<types::global_dof_index, Point<dim>> dof_location_map = 
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler);

    // Points to constraint
    Point<dim> Point1;
    Point1[0] = 0.0; Point1[1] = 0.0;

    Point<dim> Point2;
    Point2[0] = 0.0; Point2[1] = parameters.scale;

    Point<dim> Point3;
    if (dim == 3)
    {
    Point3[0] = 0.0; Point3[1] = parameters.scale; Point3[2] = parameters.scale;
    }

    const double tolerance = 1e-10;

    for (const auto &entry : dof_location_map)
    {

        const types::global_dof_index dof_index = entry.first;
        const Point<dim> &dof_location = entry.second;

        if (dof_location.distance(Point1) < tolerance)
        {
            std::cout << "Found Point 1" << std::endl;
            if (dof_index % dim == 0)
            {
                fixed_dof1_x = dof_index;
                fixed_dofs.push_back(fixed_dof1_x);
                std::cout << "Fixed X. DOF: " << fixed_dof1_x << std::endl;

            }
            if (dof_index % dim == 1)
            {
                fixed_dof1_y = dof_index;
                fixed_dofs.push_back(fixed_dof1_y);
                std::cout << "Fixed Y. DOF: " << fixed_dof1_y << std::endl;

            }
            if ((dim == 3) && (dof_index % dim == 2))
            {
                fixed_dof1_z = dof_index;
                fixed_dofs.push_back(fixed_dof1_z);
                std::cout << "Fixed Z. DOF: " << fixed_dof1_z << std::endl;
            }
        }

        if (dof_location.distance(Point2) < tolerance)
        {
            std::cout << "Found Point 2" << std::endl;
            if (dof_index % dim == 0)
            {
                fixed_dof2_x = dof_index;
                fixed_dofs.push_back(fixed_dof2_x);
                std::cout << "Fixed X. DOF: " << fixed_dof2_x << std::endl;

            }
            if ((dim == 3) && (dof_index % dim == 1))
            {
                fixed_dof2_y = dof_index;
                fixed_dofs.push_back(fixed_dof2_y);
                std::cout << "Fixed Y. DOF: " << fixed_dof2_y << std::endl;
            }
        }

        if ((dof_location.distance(Point3) < tolerance) && (dim == 3))
        {
            std::cout << "Found Point 3" << std::endl;
            if (dof_index % dim == 0)
            {
                fixed_dof3_x = dof_index;
                fixed_dofs.push_back(fixed_dof3_x);
                std::cout << "Fixed X. DOF: " << fixed_dof3_x << std::endl;
            }
        }
            
    }

    if(parameters.file_output_mode)
    {
        const std::map<dealii::types::global_dof_index, dealii::Point<dim>>
            dof_location_map = dealii::DoFTools::map_dofs_to_support_points(
                dealii::MappingQ1<dim>(), dof_handler);
        std::ofstream dof_location_file("dof_numbering.gnuplot");
        dealii::DoFTools::write_gnuplot_dof_support_point_info(dof_location_file, dof_location_map);
        std::cout << "Wrote DoF support point info to dof_numbering.gnuplot" << std::endl;
    }
}

template <int dim>
struct PointLess
{
  bool operator()(const dealii::Point<dim> &a,
                  const dealii::Point<dim> &b) const
  {
    for (unsigned int d = 0; d < dim; ++d)
    {
      if (a[d] < b[d] - 1e-14) return true;   // a  <  b
      if (a[d] > b[d] + 1e-14) return false;  // a  >  b
    }
    return false;                             // a == b
  }
};


template <int dim>
void RVE_SF<dim>::apply_boundary_conditions()
{
    
    std::cout << "Applying periodic boundary conditions with built-in applied strain..." << std::endl;
    constraints.clear();

    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);

    if (time.current() == 0)
    {
        Fbar[0][0] += 0.0;
    }
    else
    {
        Fbar[0][0] += 0.0025;
        Fbar[1][1] -= 0.00125;
        Fbar[2][2] -= 0.00125;
    }
    

    const double tol = 1e-12; // Tolerance for floating point comparison

        

    std::map<Point<dim>,
            std::array<types::global_dof_index, dim>,
            PointLess<dim>> point_to_dof;


    for (types::global_dof_index gdof = 0; gdof < dof_handler.n_dofs(); ++gdof)
    {
        /* local shape‑function index 0 … fe.n_dofs_per_cell()-1 */
        const unsigned int local_index = gdof % fe.n_dofs_per_cell();

        /* component (0 = x, 1 = y) of that local shape function */
        const unsigned int comp = fe.system_to_component_index(local_index).first;

        point_to_dof[support_points[gdof]][comp] = gdof;
    }


        /* --- lambda that imposes affine periodicity ------------------- */
        auto match_periodic = [&](const std::vector<unsigned int> &side1,
                                const std::vector<unsigned int> &side2,
                                const unsigned int periodic_coord) // 0=x,1=y
        {
            for (unsigned int i1 : side1)
            {
                const Point<dim> &p1 = support_points[i1];

                for (unsigned int i2 : side2)
                {
                    const Point<dim> &p2 = support_points[i2];

                    bool partner = true;
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        if (d == periodic_coord) continue;
                        if (std::abs(p1[d] - p2[d]) > tol)
                        {
                            partner = false;
                            break;
                        }
                    }
                    if (!partner) continue;

                    for (unsigned int comp = 0; comp < dim; ++comp)
                    {
                        const auto dof1 = point_to_dof[p1][comp];   // master
                        const auto dof2 = point_to_dof[p2][comp];   // slave

                        constraints.add_line(dof1);
                        constraints.add_entry(dof1, dof2, 1.0);

                        const double rhs =
                            ((Fbar - unit_symmetric_tensor<dim>()) * 
                            (p1 - p2))[comp];
                        constraints.set_inhomogeneity(dof1, rhs);

                        /*
                        if (parameters.terminal_output_mode)
                            std::cout << "Periodic pair: point_m = "
                                    << p1 << ", point_s = " << p2
                                    << ", comp = " << comp
                                    << ", rhs = " << rhs << std::endl;
                                    */
                    }

                    break;      // partner found → exit inner loop
                }
            }
        };



    match_periodic(left_dofs, right_dofs, 0);   // x-direction periodicity (coord=0)
    match_periodic(bottom_dofs, top_dofs, 1);   // y-direction periodicity (coord=1)
    if (dim == 3) match_periodic(back_dofs, front_dofs, 2);   // z-direction periodicity (coord=2)


    for (unsigned int &dof : fixed_dofs)
    {
        constraints.add_line(dof);   // fix u_x at (0,0)
        constraints.set_inhomogeneity(dof, 0.0);
    }

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
    /*keep_constrained_dofs=*/true);
    sparsity_pattern.copy_from(dsp);

    tangent_matrix.reinit(sparsity_pattern);
    system_rhs.reinit(dof_handler.n_dofs());
    tangent_matrix = 0.0;
    system_rhs    = 0.0;


    constraints.distribute(solution);
}


template <int dim>
void RVE_SF<dim>::assemble_system()
{

    // Output constraints to a text file for debugging
    std::ofstream constraint_file("constraints.txt");
    constraints.print(constraint_file);
    constraint_file.close();

    if(parameters.file_output_mode)
    {
        const std::map<dealii::types::global_dof_index, dealii::Point<dim>>
            dof_location_map = dealii::DoFTools::map_dofs_to_support_points(
                dealii::MappingQ1<dim>(), dof_handler);
        std::ofstream dof_location_file("dof_numbering.gnuplot");
        dealii::DoFTools::write_gnuplot_dof_support_point_info(dof_location_file, dof_location_map);
    }

    
    tangent_matrix = 0.0;
    system_rhs = 0.0;

    MappingQEulerian<dim,Vector<double>> mapping(parameters.poly_degree, dof_handler, u_total);

    FEValues<dim> fe_values(mapping, fe, quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature.size();
    const FEValuesExtractors::Vector displacements(0);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_matrix = 0.0;
        cell_rhs = 0.0;
        fe_values.reinit(cell);

        std::vector<Tensor<2,dim>> local_solution_gradients(n_q_points);
        fe_values[displacements].get_function_gradients(solution, local_solution_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            Tensor<2, dim> u_grad = local_solution_gradients[q];
            Tensor<2, dim> F = Physics::Elasticity::Kinematics::F(u_grad);

            const double Jcrit = 0.0;                     // keep it positive
            if (determinant(F) <= Jcrit)
            throw ExcMessage("detF too small");

            material.update_material(F);

            // Material Stress and tangent
            const Tensor<2, dim> P = material.compute_P(); //material.analytical_P(); 
            const Tensor<4, dim> mat_tangent = material.compute_tangent(); //material.analytical_mat(); 


           for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
            const unsigned comp_i   = fe.system_to_component_index(i).first;
            const Tensor<1,dim> grad_phi_i = fe_values.shape_grad(i,q);

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                const unsigned comp_j = fe.system_to_component_index(j).first;
                const Tensor<1,dim> grad_phi_j = fe_values.shape_grad(j,q);

                double K_ij = 0.0;
                for (unsigned int A = 0; A < dim; ++A)
                for (unsigned int C = 0; C < dim; ++C)
                    K_ij += mat_tangent[A][comp_i][C][comp_j] *
                            grad_phi_i[A] * grad_phi_j[C];

                cell_matrix(i,j) += K_ij * fe_values.JxW(q);
            }

            double R_i = 0.0;
            for (unsigned int A = 0; A < dim; ++A)
                R_i += P[A][comp_i] * grad_phi_i[A];

            cell_rhs(i) -= R_i * fe_values.JxW(q);
            }

        }
        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, tangent_matrix, system_rhs);
    }

    if(parameters.file_output_mode)
    {
    dealii::MatrixOut matrix_out;

    matrix_out.build_patches(tangent_matrix, "TangentMatrix");
    std::ostringstream fname;
    fname << "tangent_matrix_" << time.current() << ".vtk";
    std::ofstream matrix_file(fname.str());

    matrix_out.write_vtk(matrix_file);
    

    std::ostringstream fname_2;
    fname_2 << "tangent_matrix_" << time.get_timestep() << ".txt";
    std::ofstream matrix_out_txt(fname_2.str());
    for (unsigned int i = 0; i < tangent_matrix.m(); ++i)
        for (dealii::SparseMatrix<double>::const_iterator it = tangent_matrix.begin(i); it != tangent_matrix.end(i); ++it)
            matrix_out_txt << it->row() << " " << it->column() << " " << it->value() << "\n";

    // Export RHS vector (one file per time step)
    std::ostringstream fname_3;
    fname_3 << "system_rhs_" << time.get_timestep() << ".txt";
    std::ofstream rhs_out(fname_3.str());
    for (unsigned int i = 0; i < system_rhs.size(); ++i)
        rhs_out << system_rhs[i] << "\n";
    }

    if(parameters.terminal_output_mode)
    {
    bool found_nonfinite = false;
    for (unsigned int i=0; i<tangent_matrix.m(); ++i)
        for (auto it = tangent_matrix.begin(i); it != tangent_matrix.end(i); ++it)
            if (!std::isfinite(it->value())) {
                std::cout << "Non-finite matrix entry at (" << it->row() << ", " << it->column() << "): " << it->value() << std::endl;
                found_nonfinite = true;
            }
    for (unsigned int i=0; i<system_rhs.size(); ++i)
        if (!std::isfinite(system_rhs[i])) {
            std::cout << "Non-finite RHS entry at " << i << ": " << system_rhs[i] << std::endl;
            found_nonfinite = true;
        }
    if (found_nonfinite)
        std::cout << "ERROR: Non-finite entries detected in system matrix or RHS!" << std::endl;
    }
}
    #include <deal.II/lac/sparse_direct.h>
    #include <deal.II/lac/precondition.h>

template <int dim>
void RVE_SF<dim>::solve_linear_system()
{
    std::cout << "Solving linear system (" << parameters.type_lin << ")\n";

    // Incremental update Δu
    Vector<double> solution_increment(dof_handler.n_dofs());

    if (parameters.type_lin == "Direct")
    {
        // Direct LU factorization
        SparseDirectUMFPACK direct;
        direct.initialize(tangent_matrix);
        direct.vmult(solution_increment, system_rhs);
    }
    else if (parameters.type_lin == "CG")
    {
        // Conjugate Gradient (SPD only)
        SolverControl control(dof_handler.n_dofs() * parameters.max_iterations_lin,
                              parameters.tol_lin * system_rhs.l2_norm());
        SolverCG<Vector<double>> cg(control);

        if (parameters.preconditioner_type == "ssor")
        {
            PreconditionSSOR<SparseMatrix<double>> prec;
            prec.initialize(tangent_matrix, parameters.preconditioner_relaxation);
            cg.solve(tangent_matrix, solution_increment, system_rhs, prec);
        }
        else  // jacobi
        {
            PreconditionJacobi<SparseMatrix<double>> prec;
            prec.initialize(tangent_matrix, parameters.preconditioner_relaxation);
            cg.solve(tangent_matrix, solution_increment, system_rhs, prec);
        }

        std::cout << " - CG took " << control.last_step() << " iterations\n";
    }
    else if (parameters.type_lin == "GMRES")
    {
        const double r0 = system_rhs.l2_norm();
        const double tol_abs = parameters.tol_lin * r0;
        std::cout << "   GMRES: initial |r| = " << r0
                << ", tol_abs = " << tol_abs << std::endl;

   const unsigned int max_steps =
     dof_handler.n_dofs() * parameters.max_iterations_lin;
   // abs_tol = 0, rel_tol = tol_lin
   SolverControl control(max_steps,
                         /*abs_tol=*/0.0,
                         /*rel_tol=*/parameters.tol_lin);
        SolverGMRES<Vector<double>> gmres(control);

        PreconditionSSOR<SparseMatrix<double>> prec;
        prec.initialize(tangent_matrix, parameters.preconditioner_relaxation);

        gmres.solve(tangent_matrix, solution_increment, system_rhs, prec);
        std::cout << " - GMRES took " << control.last_step() << " iterations\n";
    }
    else  
    {
        SolverControl control(dof_handler.n_dofs() * parameters.max_iterations_lin,
                      parameters.tol_lin * system_rhs.l2_norm());
        SolverMinRes<Vector<double>> minres(control);

        PreconditionIdentity preconditioner;
        minres.solve(tangent_matrix, solution_increment, system_rhs, preconditioner);
    }

    
    constraints.distribute(solution_increment);
    solution += solution_increment;
}

template <int dim>
void RVE_SF<dim>::solve_nonlinear()
{
    std::cout << "Starting Nonlinear Solver" << std::endl;

    const double c_armijo = 1e-4;   // Armijo slope‑fraction
    const double beta     = 0.5;    // back‑tracking factor (α ← β α)

    unsigned int iteration = 0;

    while (iteration < parameters.max_iterations_NR)
    {
        std::cout << " - Newton iteration " << iteration << std::endl;

        

        /* residual & tangent at current guess ------------------------ */
        assemble_system();
        const double R0 = system_rhs.l2_norm();
        std::cout << "   - Residual norm: " << R0 << std::endl;

        if (R0 < parameters.tol_f)                      // ‖R‖ small enough
        {
            std::cout << "   - Converged: force residual criterion\n";
            return;
        }

        /* keep current solution ------------------------------------- */
        Vector<double> old_solution = solution;

        /* solve  K Δu = -R  (solve_linear_system adds full Δu) ------- */
        solve_linear_system();

        /* store full increment -------------------------------------- */
        Vector<double> du = solution;
        du -= old_solution;

        /* line search ----------------------------------------------- */
        double alpha = 1.0;
        bool   accepted = false;

        for (unsigned int ls = 0; ls < 20; ++ls)        // max 10 trials
        {
            /* trial solution = old + α Δu */
            solution  = old_solution;
            solution.add(alpha, du);
            constraints.distribute(solution);           // enforce BCs

            try
            {
                assemble_system();                      // R( trial )
            }
            catch (const dealii::ExceptionBase &)
            {
                /* element inverted (J≤0) → shrink step */
                alpha *= beta;
                continue;
            }

            const double R_trial = system_rhs.l2_norm();

            /* Armijo condition: sufficient decrease */
            if (R_trial <= (1.0 - c_armijo * alpha) * R0)
            {
                accepted = true;
                std::cout << "   - step accepted: α = " << alpha
                          << " , |R| = " << R_trial << std::endl;
                break;
            }

            alpha *= beta;                             // cut step
        }

        if (!accepted)
        {
            std::cout << "   - Line search failed → abort\n";
            break;
        }

        /* update norm for displacement criterion -------------------- */
        const double update_norm = alpha * du.l2_norm();
        std::cout << "   - Newton update norm: " << update_norm << std::endl;

        if (update_norm < parameters.tol_u)
        {
            std::cout << "   - Converged: displacement criterion\n";
            return;
        }




        ++iteration;
    }

    std::cout << "Warning: Nonlinear solver did not converge in "
              << parameters.max_iterations_NR << " iterations!\n";
}

template<int dim>
void RVE_SF<dim>::move_mesh()
{
    std::cout << "Moving Mesh to reflect deformation" << std::endl;

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto &cell : dof_handler.active_cell_iterators())
        for( const auto v : cell->vertex_indices())
            if (vertex_touched[cell->vertex_index(v)] == false)
            {
                vertex_touched[cell->vertex_index(v)] = true;

                Point<dim> vertex_displacement;
                for (unsigned int d = 0; d < dim; ++d)
                    vertex_displacement[d] = solution(cell->vertex_dof_index(v,d));
                
                cell->vertex(v) += vertex_displacement;
            }
}

template <int dim>
void RVE_SF<dim>::output_results()
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  /* nodal displacement ------------------------------------------------ */
  std::vector<std::string> names(dim);
  for (unsigned int d=0; d<dim; ++d) names[d]="u_"+Utilities::int_to_string(d,1);
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      inter(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, inter);

  /* cell‑wise tensor components --------------------------------------- */
  const unsigned int n_cells = triangulation.n_active_cells();
  Vector<double> F_xx(n_cells), F_xy(n_cells), F_yx(n_cells), F_yy(n_cells);
  Vector<double> P_xx(n_cells), P_xy(n_cells), P_yx(n_cells), P_yy(n_cells);

  const FEValuesExtractors::Vector disp(0);
  MappingQEulerian<dim,Vector<double>> mapping(parameters.poly_degree, dof_handler, u_total);

  FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients);

  unsigned int c = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    std::vector<Tensor<2,dim>> grad_u(quadrature.size());
    fe_values[disp].get_function_gradients(solution, grad_u);

    const Tensor<2,dim> Fq = Physics::Elasticity::Kinematics::F(grad_u[0]);
    material.update_material(const_cast<Tensor<2,dim>&>(Fq));
    const Tensor<2,dim> Pq = material.compute_P();

    F_xx[c]=Fq[0][0]; F_xy[c]=Fq[0][1];
    F_yx[c]=Fq[1][0]; F_yy[c]=Fq[1][1];
    P_xx[c]=Pq[0][0]; P_xy[c]=Pq[0][1];
    P_yx[c]=Pq[1][0]; P_yy[c]=Pq[1][1];
    ++c;
  }

  data_out.add_data_vector(F_xx,"F_xx",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(F_xy,"F_xy",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(F_yx,"F_yx",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(F_yy,"F_yy",DataOut<dim>::type_cell_data);

  data_out.add_data_vector(P_xx,"P_xx",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(P_xy,"P_xy",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(P_yx,"P_yx",DataOut<dim>::type_cell_data);
  data_out.add_data_vector(P_yy,"P_yy",DataOut<dim>::type_cell_data);

  /* write the .vtu ----------------------------------------------------- */
  data_out.build_patches(parameters.poly_degree);
  const std::string fname =
      "solution-" + Utilities::int_to_string(time.get_timestep(),4) + ".vtu";
  std::ofstream f(fname);
  data_out.write_vtu(f);
  std::cout << "Wrote " << fname << '\n';
}


template <int dim>
Tensor<2, dim> RVE_SF<dim>::compute_avg_deformation()
{
    double v0 = 0;
    Tensor<2, dim> FM;

    unsigned int n_q_size = quadrature.size();

    const FEValuesExtractors::Vector displacement(0);
    MappingQEulerian<dim,Vector<double>> mapping(parameters.poly_degree, dof_handler, u_total);

    FEValues<dim> fe_values(mapping,fe, quadrature, update_gradients | update_JxW_values | update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

                std::vector<Tensor<2,dim>> grad_u(quadrature.size());
                fe_values[displacement].get_function_gradients(solution, grad_u);

            for (unsigned int q = 0; q < n_q_size; ++q)
            {
                const Tensor<2,dim> F_q =
                    Physics::Elasticity::Kinematics::F(grad_u[q]);    

                const double dV = fe_values.JxW(q); 
                FM += F_q * dV;
                v0 += dV;
            }
        }
    
    FM /= v0;
    return FM;
}

template <int dim>
Tensor<2, dim> RVE_SF<dim>::compute_avg_stress()
{
    double v0 = 0;
    Tensor<2, dim> PM;

    const FEValuesExtractors::Vector displacement(0);
    MappingQEulerian<dim,Vector<double>> mapping(parameters.poly_degree, dof_handler, u_total);

    FEValues<dim> fe_values(mapping, fe, quadrature, update_gradients | update_JxW_values | update_quadrature_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

                std::vector<Tensor<2,dim>> grad_u(quadrature.size());
                fe_values[displacement].get_function_gradients(solution, grad_u);

            for (const unsigned int q_point : fe_values.quadrature_point_indices())
            {
                const Tensor<2,dim> F_q =
                    Physics::Elasticity::Kinematics::F(grad_u[q_point]); 
                material.update_material(const_cast<Tensor<2,dim>&>(F_q)); 
                const Tensor<2,dim> P_q = material.analytical_P();            

                const double dV = fe_values.JxW(q_point); 
                PM += P_q * dV;
                v0 += dV;
            }
        }
    
    PM /= v0;
    return PM;
}


}; // Name space



int main()
{

    using namespace RVE;
    RVE_SF<3> rve("parameters.prm");
    rve.run();
    return 0;
}