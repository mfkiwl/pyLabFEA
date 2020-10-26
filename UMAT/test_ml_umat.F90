!DIR$ FREEFORM

implicit none
character(len=80) :: cmname                                                 !User defined material name
integer, parameter :: ndi = 3                                                             !Number of direct stress components
integer, parameter :: nshr = 3                                                           !Number of engineering shear stress components
integer, parameter :: ntens = 6                                                             !Size of the stress/strain array (ndi + nshr)
integer, parameter :: nstatv = 10                                                          !Number state variables
integer, parameter :: nprops = 4                                                       !Number of user defined material constants
integer, parameter :: layer =1                                                            !Layer number
integer, parameter :: kspt =1                                                            !Section point number within the current layer
integer, parameter :: kstep =1                                                           !Step number
integer, parameter :: noel =1                                                             !Element number
integer, parameter :: npt =1                                                             !Integration point number
integer :: kinc                                                            !Increment number

real(8), parameter :: drpldt =0                                                          !Variation of rpl w.r.t. the temperature
real(8), parameter :: dtime =1.e-3                                                           !Time increment
real(8), parameter :: temp = 300                                                            !Temperature at the start of the increment
real(8), parameter :: dtemp=0                                                            !Increment of temperature.
real(8), parameter :: celent=1                                                           !Characteristic element length
real(8), parameter :: sse=1                                                              !Specific elastic strain energy
real(8), parameter :: spd=1.e-3                                                              !Specific plastic dissipation
real(8), parameter :: scd=1.e-3                                                              !Specific creep dissipation
real(8), parameter :: rpl=1.e-3                                                              !volumetric heat generation per unit time
real(8), parameter :: pnewdt =1                                                          !Ratio dtime_next/dtime_now

real(8) :: stress(ntens)                                                    !Stress tensor
real(8) :: ddsdde(ntens,ntens)                                              !Jacobian matrix of the constitutive model
real(8) :: ddsddt(ntens)                                                    !Variation of the stress increment w.r.t. to the temperature
real(8) :: drplde(ntens)                                                    !Variation of rpl w.r.t. the strain increment
real(8) :: stran (ntens)                                                    !Total strain tensor at the beggining of the increment
real(8) :: dstran(ntens)                                                    !Strain increment

real(8) :: statev(nstatv)                                                   !Solution dependent state variables
real(8) :: props (nprops)                                                   !User specified material constants 
real(8) :: dfgrd0(3,3)                                                      !Deformation gradient at the beggining of the increment
real(8) :: dfgrd1(3,3)                                                      !Deformation gradient at the end of the increment
real(8) :: drot  (3,3)                                                      !Rotation increment matrix
real(8) :: coords(3)                                                        !Coordinates of the material point
real(8) :: time  (2)                                                        !1: Step time; 2:Total time; Both at the beggining of the inc
real(8) :: predef(1)                                                        !Predefined field variables at the beggining of the increment
real(8) :: dpred (1)    

props(1) = 200000.     ! Young's modulus
props(2) = 0.3         ! Poisson ratio
props(3) = 150.        ! yield strength
props(4) = 1.          ! Work hardening rate

open(30,file='test-output.dat', status='unknown', position='rewind')
write(30,*)'strain (6-array) stress (6-array)'
stress = 0.d0
stran = 0.d0
dstran = 0.d0
dstran(1) = 0.0003d0
dstran(2) = -0.3*dstran(1)
dstran(3) = -0.3*dstran(1)

do kinc=1,5
   call umat(stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt,drplde,     &
 &               drpldt, stran, dstran, time, dtime, temp, dtemp, predef,       &
 &               dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops,        & 
 &               coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt,       &
 &               layer, kspt, kstep, kinc)

    write(30,*)  stran, stress
end do
close(30)
10 format('# ',A20,6G15.3)

end

!End MAIN
!=========================================================================
 subroutine umat(stress, statev, ddsdde, sse, spd, scd, rpl, ddsddt,drplde,     &
 &               drpldt, stran, dstran, time, dtime, temp, dtemp, predef,       &
 &               dpred, cmname, ndi, nshr, ntens, nstatv, props, nprops,        & 
 &               coords, drot, pnewdt, celent, dfgrd0, dfgrd1, noel, npt,       &
 &               layer, kspt, kstep, kinc)
 !========================================================================
 ! variables available in UMAT: 
 !------------------------------
 ! stress, stran, statev, props
 ! dstran, drot, dfgrd0, dfgrd1, time, dtime, temp, dtemp, predef, dpred
 ! coords, celent
 ! number of element, integration point, current step and increment
 !--------------------------------
 ! variables that must be defined:
 !--------------------------------
 ! stress, statev, ddsdde
 !==========================================================================
 ! Material properties (props):
 ! 1: E-modulus
 ! 2: Poisson ratio
 ! 3: yield strength
 ! 4: linear strain hardening coefficient
 !==========================================================================
 ! State variables
 ! 1-6: plastic strain (eplas)
 
    implicit none
 
    character(len=80) :: cmname                                                 !User defined material name
    integer :: ndi                                                              !Number of direct stress components
    integer :: nshr                                                             !Number of engineering shear stress components
    integer :: ntens                                                            !Size of the stress/strain array (ndi + nshr)
    integer :: nstatv                                                           !Number state variables
    integer :: nprops                                                           !Number of user defined material constants
    integer :: layer                                                            !Layer number
    integer :: kspt                                                             !Section point number within the current layer
    integer :: kstep                                                            !Step number
    integer :: noel                                                             !Element number
    integer :: npt                                                              !Integration point number
    integer :: kinc                                                             !Increment number
 
    real(8) :: drpldt                                                           !Variation of rpl w.r.t. the temperature
    real(8) :: dtime                                                            !Time increment
    real(8) :: temp                                                             !Temperature at the start of the increment
    real(8) :: dtemp                                                            !Increment of temperature.
    real(8) :: celent                                                           !Characteristic element length
    real(8) :: sse                                                              !Specific elastic strain energy
    real(8) :: spd                                                              !Specific plastic dissipation
    real(8) :: scd                                                              !Specific creep dissipation
    real(8) :: rpl                                                              !volumetric heat generation per unit time
    real(8) :: pnewdt                                                           !Ratio dtime_next/dtime_now
 
    real(8) :: stress(ntens), dsig(ntens)                                       !Stress tensor at start of increment, stress increment
    real(8) :: ddsdde(ntens,ntens)                                              !Jacobian matrix of the constitutive model
    real(8) :: ddsddt(ntens)                                                    !Variation of the stress increment w.r.t. to the temperature
    real(8) :: drplde(ntens)                                                    !Variation of rpl w.r.t. the strain increment
    real(8) :: stran (ntens)                                                    !Total strain tensor at the beggining of the increment
    real(8) :: dstran(ntens)                                                    !Strain increment
    real(8) :: eplas(ntens)                                                     !plastic strain
 
    real(8) :: statev(nstatv)                                                   !Solution dependent state variables
    real(8) :: props (nprops)                                                   !User specified material constants 
    real(8) :: dfgrd0(3,3)                                                      !Deformation gradient at the beggining of the increment
    real(8) :: dfgrd1(3,3)                                                      !Deformation gradient at the end of the increment
    real(8) :: drot(3,3)                                                      !Rotation increment matrix
    real(8) :: coords(3)                                                        !Coordinates of the material point
    real(8) :: time(2)                                                        !1: Step time; 2:Total time; Both at the beggining of the inc
    real(8) :: predef(1)                                                        !Predefined field variables at the beggining of the increment
    real(8) :: dpred (1)                                                        !Increment of predefined field variables
 
    character(200) :: filePath, supportVectorsPath, dualCoeffsPath, fileName    !cwd path and files containing the support vectors and dual coeffs
    real(8) :: ymod, nu, ebulk3, eg2, eg, eg3, elam                             !Lamé parameters and derived coefficients                                           
    integer, parameter :: nsv = 125 !1689                                       !Number of support vectors and dual coefficients (To do: pass via props)    
    integer, parameter :: nsc = 3                                               ! Number of components in cylindrical stress tensor
    real(8), dimension(nsv, 2) :: supportVectors                                !Array of support vectors
    real(8), dimension(nsv) :: dual_coeffs                                      !Array of dual coefficients
    real(8), dimension(nsc) :: sigmaCyl, sc0, sc1                               !Stress vector in the cylindrical coordinate system
    real(8), dimension(ntens) :: stress_fl                                      ! flow stress on yield locus
    real(8), dimension(3) :: a_vec, b_vec                                       !vectors a_vec and b_vec are the spanning vectors of the deviatoric plane 
    real(8) :: rho                                                              !Intercept of the decision rule function
    real(8) :: lambda                                                           !Regularization parameter
    real(8) :: scale_seq                                                        ! scaling factor for stress
    real(8) :: fsvc                                                             !Decision function
    real(8), dimension(3) :: dfds                                            !Derivative of the decision function w.r.t. princ. stress
    real(8), dimension(3,3) :: jac                                              !Jacobian of the coordinate transformation
    real(8), dimension(3) :: flow                                             !Flow vector
 
    real(8), parameter :: tol = 1.0e-03                                         !Tolerance for the fsvc root finder
    integer, parameter :: nmax = 100                                            !Maximum number of iterations for the fsvc root finder
 
    integer :: i, j, fileUnit, niter                                                   !Auxiliar indices
    real(8), dimension(2, nsv) :: temp_sv                                       !Auxiliary array
    real(8), dimension(ntens, ntens) :: Ct                                 !Consistent tanget stiffness matrix
    real(8), dimension(ntens) :: deps                                           !strain increment outside yield locus, if load step is splitted
 
    real(8) :: threshold, pi, h1, h2, eeq, peeq, sc_elstep
 
    pi = 4.d0*datan(1.d0)
    !The working directory path is defined, along with the basename of the files
    !containing the support vectors and dual coefficients
    filePath ='./data/'
    fileName = 'umat-output.dat'
    
    supportVectorsPath = trim(adjustl(filePath))//'supportVectors.out'
 
    !dualCoeffsPath = trim(adjustl(filePath))//'dualCoeffsTresca.out'
    dualCoeffsPath = trim(adjustl(filePath))//'dualCoeffs.out'
 
    open(newunit=fileUnit, file=supportVectorsPath, status='old')
    read(fileUnit, *) temp_sv
    close(fileUnit)
    supportVectors = transpose(temp_sv)
 
    open(newunit=fileUnit, file=dualCoeffsPath, status='old')
    read(fileUnit, *) dual_coeffs
    close(fileUnit)
 
    !Read from the python script (Hill)
    rho = 0.54039174d0
 
    ! gamma parameter (Tresca)
    !lambda = 0.0004
 
    ! gamma parameter (Hill)
    lambda = 4.0d0
    
    scale_seq = 150.d0
    threshold = 0.5d0   ! threshold value for yield function still accepted as elastic
 
    !Definition of the Lamé parameters used to fill out the stiffness matrix
    ymod = props(1)
    nu = props(2)
    ebulk3 = ymod/(1.-2.*nu)
    eg2 = ymod/(1.+nu)
    eg = eg2/2.d0
    eg3 = 3.d0*eg
    elam = (ebulk3 - eg2)/3.d0
    
    !get accumulated plastic strain
    eplas = statev(1:6)
    
    ! a_vec and b_vec are the spanning vectors of the deviatoric plane
    h1 = 1./sqrt(6.d0)
    h2 = 1./sqrt(2.d0)
    a_vec(1) = 2.d0*h1
    a_vec(2) = -h1
    a_vec(3) = -h1
    b_vec(1) = 0.d0
    b_vec(2) = h2
    b_vec(3) = -h2

 
    !Elastic stiffness matrix is defined
    !isotropic elasticity, plane strain conditions
    ddsdde = 0.d0
    sc_elstep = 0.d0
    do i=1, ndi
        do j=1, ndi
            ddsdde(i,j)=elam
        end do
        ddsdde(i,i)=eg2+elam
    end do
    do i=ndi+1, ntens
        ddsdde(i,i)=eg
    end do
    Ct = ddsdde
    deps = dstran
 
    !The trial elastic stress is computed
    dsig = 0.d0
    do i=1, ntens
        do j=1, ntens
            dsig(i)=dsig(i)+ddsdde(i,j)*dstran(j)
        end do
    end do
    
     ! calculate yield function and cylindrical stress (equiv. stress, polar angle, hydrostatic stress)
     call calcFSVC(stress+dsig, sigmaCyl, 0, fsvc)
  
     niter = 0
     do while ((fsvc.ge.threshold).and.(niter<5))
         ! stress lies outside yield locus
         ! 1. calculate proper flow stress on yield locus
         sc1 = sigmaCyl
         call findRoot(sigmaCyl, stress_fl, fsvc)
         call calcFSVC(stress, sc0, 0, h1)    ! cylindrical stress at start of increment
         if (h1.lt.-tol) then
            !load step started in elastic regime and has to be splited
            !for load reversals, negative values must be treated separately
            !perform elastic substep
            sc_elstep = (sigmaCyl(1)-sc0(1))/(sc1(1)-sc0(1))
            deps = dstran*sc_elstep
            dsig = 0.d0
            do i=1, ntens
                do j=1, ntens
                    dsig(i)=dsig(i)+ddsdde(i,j)*deps(j)
                end do
            end do
            call calcFSVC(stress+dsig, sigmaCyl, 0, fsvc)
            if (fsvc.ge.threshold) then
                print*,'***Warning: Splitting of load step failed', fsvc, sc0, sigmaCyl
                print*,'DSTRAN, deps', dstran, deps
                print*,'stress, dsig', stress, dsig
            else
                ! update internal varialbles and  perform plastic part of load increment
                stran = stran + deps
                stress = stress + dsig
                deps = dstran - deps
                dsig = 0.d0
                do i=1, ntens
                    do j=1, ntens
                        dsig(i)=dsig(i)+ddsdde(i,j)*deps(j)
                    end do
                end do
            end if
         end if
         ! 2. calculate gradient on yield locus for given cyl. stress tensor
         call calcDfsvcds(sigmaCyl, stress_fl, dfds)
         ! 3. calculate plastic strain increment 'flow'
         call calcFlow(dfds, deps, Ct, eplas, flow)
         ! 4. calculate consistent tangent stiffness tensor
         call calcTangstiff(ddsdde, dfds, Ct)
         ! 5. calculate consistent stress increment
         dsig = 0.
         do i=1, ntens
             do j=1, ntens
                 dsig(i)=dsig(i)+Ct(i,j)*deps(j)
             end do
         end do
         ! update stress for next iteration and
         ! calculate yield function
         call calcFSVC(stress+dsig, sigmaCyl, 0, fsvc)
     end do
     if (niter.ge.5) then
        print*,'***Warning: plasticity algorithm did not converge after',niter,'iterations'
    end if
        
    !update stresses and strain
    stran = stran + deps
    stress = stress + dsig
    eplas = eplas + flow 
    !update internal variables
    statev(1:6) = eplas
    !update material Jacobian
    ddsdde = ddsdde*sc_elstep + Ct*(1.d0-sc_elstep)
    
    !output
    if (kinc==1) then
        open(newunit=fileUnit,file=trim(adjustl(fileName)),    &
        &    status='replace', action='write')
        write(fileUnit,'(a)') '#strain.123 (.), stress.123 (MPa), pl. strain.123 (.), equ. stress (MPa), equ. strain (.), peeq (.)'
    else
        open(newunit=fileUnit,file=trim(adjustl(fileName)),    &
        &    status='old', position='append', action='write')
    end if
    call calcCylStress(stress, sigmaCyl)
    call calcEqStrain(stran, eeq)
    call calcEqStrain(eplas, peeq)
    write(fileUnit,'(12f14.7)') stran(1:3), stress(1:3), eplas(1:3), sigmaCyl(1), eeq, peeq
    flush(fileUnit)
    close(fileUnit)
 
 
    contains
 
 
    subroutine calcHydStress(stress, ntens, sigmaHyd)
    !This subroutine caculates the hydrostatic stress of the Cauchy stress
 
        implicit none
        integer :: ntens
        real(8), dimension(ntens) :: stress
        real(8) :: sigmaHyd
 
        sigmaHyd = (stress(1)+stress(2)+stress(3))/3.
       
    end subroutine calcHydStress
 
 
    subroutine calcDevStress(stress, sigmaDev, sigmaHyd)
    !This subroutine calculate the deviatoric stress of the Cauchy stress
 
        implicit none
        real(8), dimension(ntens) :: stress, sigmaDev
        real(8) :: sigmaHyd
        integer :: i
 
        call calcHydStress(stress, ntens, sigmaHyd)
        do i=1, ndi
            sigmaDev(i) = stress(i) - sigmaHyd
        end do
        do i=ndi+1, ntens
            sigmaDev(i) = stress(i)
        end do
    end subroutine calcDevStress
 
 
    subroutine calcThetaDev(stress, thetaDev)
    !This subroutine calculates the generalized Lode angle on the deviatoric
    !plane (Eq. 13)
 
        implicit none
        integer :: i
        real(8), dimension(ntens) :: stress
        real(8), dimension(3) :: princSigma, princDev
        real(8) :: sigmaHyd, thetaDev
 
        !The principal stresses are computed and stored in the vector princSigma
        call sprinc(stress, princSigma, 1, ndi, nshr)
 
        !The hydrostatic stress is fetched
        call calcHydStress(stress, ntens, sigmaHyd)
 
        !The deviatoric principal stresses are calculated and stored at the 
        !vector princDev
        do i=1,ndi
            princDev(i) = princSigma(i) - sigmaHyd
        end do
 
        thetaDev = atan2( dot_product(princDev, b_vec),                        &
        &                 dot_product(princDev, a_vec))
 
    end subroutine calcThetaDev
 
    !*******************************************
    subroutine sprinc(sig, princ, n, ndi, nshr)
        !*******************************************
        ! This is only a dummy for the Abaqus in-built function
        ! must be removed when used as UMAT !!!!
        !********************************************
        implicit none
        integer, parameter :: ntens = 6
        integer :: n, ndi, nshr
        real(8), dimension(ntens) :: sig
        real(8), dimension(3) :: princ
        
        princ(1) = sig(1)
        princ(2) = sig(2)
        princ(3) = sig(3)
        
    end subroutine sprinc
 
    subroutine calcCylStress(stress,sigmaCyl)
    !This subroutine calculates the stress vector in the cylindrical COS
        implicit none
        real(8), dimension(ntens) :: stress, sigmaDev
        real(8), dimension(3) :: princDev
        real(8) :: thetaDev, sigmaHyd
        real(8), dimension(nsc) :: sigmaCyl
 
        call calcDevStress(stress, sigmaDev, sigmaHyd)
        sigmaCyl(3) = sigmaHyd
 
        call sprinc(sigmaDev, princDev, 1, ndi, nshr)
 
        !The J2 equivalent stress of the vector of principal stresses of the 
        !stress deviator is computed (Eq. 3)
        sigmaCyl(1) = sqrt( 0.5*((princDev(1)-princDev(2))**2 +                &
        &                        (princDev(2)-princDev(3))**2 +                &
        &                        (princDev(3)-princDev(1))**2))
 
        call calcThetaDev(stress, thetaDev)
        sigmaCyl(2) = thetaDev
 
    end subroutine calcCylStress
    
    subroutine calcPrincStress(sigmaCyl, stress)
        ! convert cylindrical stresses into principal stresses
        implicit none
        real(8), dimension(ntens) :: stress
        real(8), dimension(nsc) :: sigmaCyl
        real(8) :: theta, hh, cs 
        integer :: i 
    
        theta = sigmaCyl(2)
        cs = sqrt(2./3.)
        stress=0.
        do i=1,ndi
            hh  = cos(theta)*a_vec(i) + sin(theta)*b_vec(i)
            stress(i) = hh*cs*sigmaCyl(1) + sigmaCyl(3)
        end do
    end subroutine calcPrincStress
    
    subroutine calcEqStrain(eps, eeq)
        !calculate equivalent strain from Voigt strain tensor
        real(8), dimension(ntens) :: eps
        real(8) :: eeq
        real(8), dimension(ntens) :: ed
        real(8) :: ev, hdi, hsh
        integer :: i
        
        ev = 0.d0
        do i=1,ndi
            ev = ev + eps(i)
        end do
        ed = eps
        hdi = 0.d0
        hsh = 0.d0
        do i=1,ndi
            hdi = hdi + (eps(i)-ev)**2
            hsh = hsh + eps(ndi+i)**2
        end do
        eeq = sqrt((2.d0*hdi+hsh)/3.)
    end subroutine calcEqStrain
 
    subroutine calcKernelFunction(sigmaCyl, supportVector, kernelFunc)
    !This subroutine calculates the Radial Basis Function (RBF) kernel of the
    !svc (Eq. 19)
        implicit none
        real(8), dimension(nsc) :: sigmaCyl
        real(8), dimension(2) :: supportVector, hs
        real(8) :: kernelFunc, hh
        
        hs(1) = sigmaCyl(1)/scale_seq - 1. - supportVector(1)
        hs(2) = sigmaCyl(2)/pi - supportVector(2)
        hh = hs(1)*hs(1) + hs(2)*hs(2)
        kernelFunc = exp(-lambda*hh)
    end subroutine calcKernelFunction
 
 
    subroutine calcDK_DS(sigmaCyl, supportVector, dk_ds)
    !This subroutine calculates the derivative of the kernel basis function
    !with respect to the cylindrical stress vector (Eq. 21)
        implicit none
        real(8), dimension(nsc) :: sigmaCyl
        real(8), dimension(2)  :: supportVector
        real(8) :: kernelFunc, dk_ds
 
        call calcKernelFunction(sigmaCyl, supportVector, kernelFunc)
        dk_ds = -2.*lambda*kernelFunc*(sigmaCyl(2) - supportVector(2))
    end subroutine calcDK_DS
 
 
    subroutine calcFSVC(stress, sigmaCyl, flag, fsvc)
    !This subroutine computes the decision function F (Eq. 18)
    !based on the trained Support Vector Classification (SVC)
        implicit none
        integer :: i
        real(8), dimension(ntens) :: stress
        real(8), dimension(nsc) :: sigmaCyl
        real(8) :: fsvc, kernelFunc
        integer :: flag
 
        if (flag.eq.0) then
           call calcCylStress(stress, sigmaCyl)
        end if
        fsvc = 0.
        do i=1, nsv
            call calcKernelFunction(sigmaCyl, supportVectors(i, :), kernelFunc)
            fsvc = fsvc + dual_coeffs(i)*kernelFunc
            !print*,i,supportVectors(i,1:2), dual_coeffs(i)
        end do
        fsvc = fsvc + rho
    end subroutine calcFSVC
 
 
    subroutine calcDfsvcds(sigmaCyl, stress, dfds)
    !This subroutine calculates the derivative of the decision function w.r.t.
    !principal stress
        implicit none
        integer :: i
        real(8), dimension(nsc) :: sigmaCyl
        real(8), dimension(ntens) :: stress
        real(8), dimension(3) :: dfds
        real(8), dimension(3,3) :: jac
        real(8) :: dfsvc, dk_ds
 
        ! calculate derivative w.r.t. cylindrical stress
        dfsvc = 0.
        do i=1, nsv
            call calcDK_DS(sigmaCyl, supportVectors(i, :), dk_ds)
            dfsvc = dfsvc + dual_coeffs(i)*dk_ds
        end do
        
        ! multiply with Jacobian to get gradient in princ. stress space
        call CalcJac(sigmaCyl, stress, jac)
        do i=1,3
            dfds(i) = jac(i,1) + jac(i,2)*dfsvc
        end do
    end subroutine calcDfsvcds
 
 
    subroutine calcJac(sigmaCyl, stress, jac)
    !This subroutine calculates the jacobian of the coordinate transformation
    !from cylindrical into princ. stress space
        implicit none
        real(8), dimension(nsc) :: sigmaCyl
        real(8), dimension(ntens) :: stress, sdev
        real(8), dimension(3,3) :: jac
        real(8) :: sigmaHyd, vn
        real(8), dimension(3) :: dseqds, dsa, dsb, sp
        complex :: sc, z
        integer :: i

        call calcDevStress(stress, sdev, sigmaHyd)
        call sprinc(stress, sp, 1, ndi, nshr)
        vn = 0.
        do i=1,ntens
            vn = vn + sdev(i)*sdev(i)
        end do
        vn = sqrt(1.5*vn)   !norm of stress vector
        if (vn.gt.0.1) then
           !only calculate Jacobian if sig>0
            dseqds = 3.*sdev(1:3)/vn
            jac(:,1) = dseqds
            dsa = 0.
            dsb = 0.
            do i=1,3
                dsa = dsa + sp(i)*a_vec(i)
                dsb = dsb + sp(i)*b_vec(i)
            end do
            do i=1,3
                sc = cmplx(dsa(i), dsb(i))
                !z = (0.,-1.)*(cmplx(a_vec(i), b_vec(i))/sc - dseqds/vn)
                z = (0., -1.)*(cmplx(a_vec(i), b_vec(i))/sc - dseqds(i)/vn)
                jac(i,2) = real(z)
            end do
            jac(:,3)=1.d0/3.d0
        end if
    end subroutine calcJac
 
 
    subroutine calcFlow(dfsvc, deps, Ct, eplas, flow)
    !This subroutine calculates the flow vector N (Eq. 6 and Eq. 16)
        implicit none
        real(8), dimension(3) :: dfsvc
        real(8), dimension(ntens) :: deps, eplas, flow
        real(8), dimension(ntens, ntens) :: Ct
        real(8) :: hh, l_dot
        integer :: i,j
 
        hh = 0.
        l_dot = 0.
        do i=1,3
            do j=1,3
                hh = hh + dfsvc(i)*Ct(i,j)*dfsvc(j)
                ! add hardening term  + 4.*self.Hpr
            end do
        end do
        do i=1,3
            do j=1,3
                l_dot = l_dot + dfsvc(i) * Ct(i,j) * deps(j) / hh  ! deps must not contain elastic strain components
            end do
        end do
        flow = l_dot * dfsvc 
    end subroutine calcFlow
 
    
    subroutine calcTangStiff(Cel, dfds, Ct)
    !This subroutine calculates the elasto-plastic tangent stiffness matrix 
    !(Eq. 9)
        implicit none
        real(8), dimension(ntens, ntens) :: Cel, Ct
        real(8), dimension(3) :: dfds
        real(8), dimension(ntens) :: ca
        real(8), dimension(ntens, ntens) :: C_temp
        real(8) :: hh
        integer :: i, j

        hh = 0.d0
        ca = 0.d0
        Ct = 0.d0
        do i=1,ntens
            do j=1,ntens
                hh =  hh + dfds(i) * Cel(i,j) * dfds(j) ! + 4.*self.Hpr
                ca(i) = ca(i) + Cel(i,j) * dfds(j)
            end do
        end do
        do i=1,ntens
            do j=1,ntens
                Ct(i,j) = Cel(i,j) - ca(i)*ca(j)/hh
            end do
        end do
    end subroutine calcTangStiff
 
 
    subroutine findRoot(sigmaCyl, s_fl, fsvc)
    !This subroutine implements the bisection method for finding the root of the
    !decision rule function
        implicit none
 
        integer :: i, j
        real(8), dimension(nsc) :: sigmaCyl
        real(8), dimension(ntens) :: s_fl
        real(8) :: fsvc, error
 
        real(8) :: lowerBound, upperBound, increment
        real(8), dimension(ntens) :: sdum
        integer, parameter :: split = 10
        real(8), dimension(2, split) :: storage
        real(8) :: a, b, fsvca, fsvcb, root, fsvcAtroot, seq0
 
        sdum = 0.
        call calcFSVC(sdum, sigmaCyl, -1, fsvca)
 
        if (fsvca .le. tol) then
            return
        else
            !An initial broad interval is split into split subintervals and a change
            !in the sign of fsvc is sought. The first subinterval meeting this 
            !criterial will be used for the bisection root finding procedure
            ! It is assumed that the fSVC value at sigmaCyl is positive turning negative at 
            ! smaller stresses
            seq0 = sigmaCyl(1)
            upperBound = seq0
            a = upperBound
            lowerBound = 0.9*seq0
            call calcFSVC(sdum, sigmaCyl, -1, fsvcb)
            
            increment = lowerBound/split
            j = 1
            do while ((fsvca*fsvcb .gt. 0.).or.(j.eq.split))
                b = lowerBound - j*increment
                sigmaCyl(1) = b
                call calcFSVC(sdum, sigmaCyl, -1, fsvcb)
                j = j + 1
            end do 
            increment = (a-b)/split
            j = 1
            sigmaCyl(1) = a
            do while ((fsvca*fsvcb .lt. 0.).or.(j.eq.split))
                a = upperBound - j*increment
                sigmaCyl(1) = a
                call calcFSVC(sdum, sigmaCyl, -1, fsvca)
                j = j + 1
            end do 
            a = a + increment
            sigmaCyl(1) = a
            call calcFSVC(sdum, sigmaCyl, -1, fsvca)
 
            i = 1
            error = 1.
            do while ((i .lt. nmax) .and. (error .ge. 0.5))
                !Calculating fsvc at the bounds of the interval
                sigmaCyl(1) = a
                call calcFSVC(sdum, sigmaCyl, -1, fsvca)
 
                sigmaCyl(1) = b
                call calcFSVC(sdum, sigmaCyl, -1, fsvcb)
 
                !Checking if the root is bracketed within the interval
                if (fsvca*fsvcb .lt. 0.) then
                    root = (a+b)/2.
                    sigmaCyl(1) = root
                    call calcFSVC(sdum, sigmaCyl, -1, fsvcAtroot)
 
                    if (fsvca*fsvcAtroot .lt. 0.) then
                        b = root
                    else
                        a = root
                    end if
                else
                    print*, 'Root not bracketed within the stipulated interval'
                    print*, a, fsvca
                    print*, b, fsvcb
                    stop
                end if
                i = i + 1
                error = abs(fsvcAtroot)
            end do
            if (abs(fsvca).lt.error) then
                sigmaCyl(1) = a
            end if
            if (abs(fsvcb).lt.error) then
                sigmaCyl(1) = b
            end if
            ! scale hydrostatic stress according to equiv. stress
            sigmaCyl(3) = sigmaCyl(3)*sigmaCyl(1)/seq0
            call calcPrincStress(sigmaCyl, s_fl)
        end if 
 
    end subroutine findRoot
 
 
    subroutine uhard(syield, hard, eqplas, eqplasrt, time, dtime, temp, dtemp, &
    &                noel, npt, layer, kspt, kstep, kinc, cmname, nstatv,      &
    &                statev, numfieldv, predef, dpred, numprops, props)
    !This subroutine provides the interface for defining the hardening behavior
    !To be adapated to the corresponding hardening definition
        implicit none
        real(8) :: syield
        real(8), dimension(3) :: hard
        real(8) :: eqplas, eqplasrt, time, dtime, temp, dtemp
        integer :: noel, npt, layer, kspt, kstep, kinc
        integer :: nstatv, numprops, numfieldv
        real(8) :: predef(numfieldv)
        real(8), dimension(nstatv) :: statev
        real(8), dimension(*) :: dpred, props
        character(len=80) :: cmname
 
        syield = props(3) + props(4)*eqplas
        hard(1) = props(4)
        hard(2) = 0.
        hard(3) = 0.
 
    end subroutine uhard
 
 end subroutine umat
    
    
    