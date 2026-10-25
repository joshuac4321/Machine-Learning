@Autonomous(name="Basic Auto", group="Robot")
public class BasicAutonomous extends LinearOpMode {
    private DcMotor leftDrive = null;
    private DcMotor rightDrive = null;

    @Override
    public void runOpMode() {
        // Step 1: Initialize the hardware variables using the hardware map
        leftDrive  = hardwareMap.get(DcMotor.class, "left_drive");
        rightDrive = hardwareMap.get(DcMotor.class, "right_drive");

        AprilTagProcessor myAprilTagProcessor =
        AprilTagProcessor.easyCreateWithDefaults();
        
        VisionPortal myVisionPortal;
        myVisionPortal = VisionPortal.easyCreateWithDefaults(
            hardwareMap.get(WebcamName.class, "Webcam 1"), 
            myAprilTagProcessor
        );

        // Step 2: Set motor directions (reverse one side so forward moves the robot straight)
        leftDrive.setDirection(DcMotor.Direction.FORWARD);
        rightDrive.setDirection(DcMotor.Direction.REVERSE);

        // Step 3: Optional telemetry to signal initialization is complete
        telemetry.addData("Status", "Initialized");
        telemetry.update();

        // Step 4: Wait for the game to start (Driver Station presses PLAY)
        waitForStart();

        // Step 5: Autonomous action code goes here after the start signal
        if (opModeIsActive()) {
            // Example: Drive forward for a bit
            leftDrive.setPower(0.5);
            rightDrive.setPower(0.5);
            sleep(1000);
            leftDrive.setPower(0);
            rightDrive.setPower(0);
        }
    }
}