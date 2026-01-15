/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker.fragment

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Point
import android.graphics.PointF
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.DngCreator
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.params.MeteringRectangle
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.SessionConfiguration
import android.media.ExifInterface
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import android.util.SizeF
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.TextureView
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.lifecycle.lifecycleScope
import androidx.navigation.Navigation
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.viewpager2.widget.ViewPager2.SCROLL_STATE_DRAGGING
import com.google.mediapipe.examples.facelandmarker.FaceLandmarkerHelper
import com.google.mediapipe.examples.facelandmarker.MainViewModel
import com.google.mediapipe.examples.facelandmarker.R
import com.google.mediapipe.examples.facelandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.Arrays
import java.util.Collections
import java.util.Comparator
import java.util.LinkedHashMap
import java.util.LinkedHashSet
import java.util.Locale
import java.util.TreeMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.math.atan
import kotlin.math.max
import kotlin.math.min

class CameraFragment : Fragment(), FaceLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Face Landmarker"
        private const val ZOOM_LEVEL_CAPTURE = 3.0f
        private const val ZOOM_LEVEL_WIDE = 1.0f
        private const val FRONT_DEFAULT_ZOOM = 1.25f
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private lateinit var faceLandmarkerHelper: FaceLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private val faceBlendshapesResultAdapter by lazy {
        FaceBlendshapesResultAdapter()
    }

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

    // Camera2 variables
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null
    private var rawImageReader: ImageReader? = null
    private var previewRequestBuilder: CaptureRequest.Builder? = null
    private var cameraId: String? = null
    private var backgroundHandler: Handler? = null
    private var backgroundThread: HandlerThread? = null
    private var previewSize: Size? = null
    private var isFrontCamera = false
    private var isFlashOn = false
    private var maxZoom: Float = 1f
    private var zoomRatioRange: Range<Float>? = null
    private var activeArraySize: Rect? = null
    @Volatile private var lastCropRegion: Rect? = null
    private var sensorOrientation: Int = 0
    private var currentCharacteristics: CameraCharacteristics? = null

    // Lens Selection
    private data class LensOption(
        val openCameraId: String,      // usually the logical camera to open
        val physicalId: String?,       // physical lens to force
        val label: String,
        val focalMm: Float?,
        val sensorArea: Int // Width * Height of active array
    )
    private var availableLenses: List<LensOption> = emptyList()
    private var selectedPhysicalId: String? = null
    private var currentCameraIndex = 0

    // Analysis
    private var isAnalyzing = false
    private var analysisBitmap: Bitmap? = null
    @Volatile private var mpInputImageWidth = 0
    @Volatile private var mpInputImageHeight = 0

    // Automated Capture
    private var isAutomatedCaptureRunning = false
    private var captureJob: Job? = null
    private var sharpnessThreshold = 20.0
    private var leftEyeOpenness: Float = 1.0f
    private var rightEyeOpenness: Float = 1.0f
    private var smoothBox: RectF? = null

    // RAW Buffer Matching
    private val rawResultQueue = TreeMap<Long, TotalCaptureResult>()
    private val rawImageQueue = TreeMap<Long, Image>()

    // Filename Matching
    private val filenameMap = ConcurrentHashMap<Long, String>()

    // Modes
    private var isRawMode = false
    private var isBurstMode = false

    // Reusable callback for repeating requests
    private val repeatingPreviewCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
            lastCropRegion = result.get(CaptureResult.SCALER_CROP_REGION)
        }
    }

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(
                requireActivity(), R.id.fragment_container
            ).navigate(R.id.action_camera_to_permissions)
            return
        }

        // Maximize brightness
        val layoutParams = requireActivity().window.attributes
        layoutParams.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_FULL
        requireActivity().window.attributes = layoutParams

        startBackgroundThread()

        // Start the FaceLandmarkerHelper again when users come back
        backgroundExecutor.execute {
            if (faceLandmarkerHelper.isClose()) {
                faceLandmarkerHelper.setupFaceLandmarker()
            }
        }

        if (fragmentCameraBinding.viewFinder.isAvailable) {
            openCamera(fragmentCameraBinding.viewFinder.width, fragmentCameraBinding.viewFinder.height)
        } else {
            fragmentCameraBinding.viewFinder.surfaceTextureListener = surfaceTextureListener
        }
    }

    override fun onPause() {
        stopAutomatedCapture()
        closeCamera()
        stopBackgroundThread()
        super.onPause()

        // Restore brightness
        val layoutParams = requireActivity().window.attributes
        layoutParams.screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE
        requireActivity().window.attributes = layoutParams

        if(this::faceLandmarkerHelper.isInitialized) {
            viewModel.setMaxFaces(faceLandmarkerHelper.maxNumFaces)
            viewModel.setMinFaceDetectionConfidence(faceLandmarkerHelper.minFaceDetectionConfidence)
            viewModel.setMinFaceTrackingConfidence(faceLandmarkerHelper.minFaceTrackingConfidence)
            viewModel.setMinFacePresenceConfidence(faceLandmarkerHelper.minFacePresenceConfidence)
            viewModel.setDelegate(faceLandmarkerHelper.currentDelegate)

            // Close the FaceLandmarkerHelper and release resources
            backgroundExecutor.execute { faceLandmarkerHelper.clearFaceLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding =
            FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        with(fragmentCameraBinding.recyclerviewResults) {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = faceBlendshapesResultAdapter
        }

        // Initialize our background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        backgroundExecutor.execute {
            faceLandmarkerHelper = FaceLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minFaceDetectionConfidence = viewModel.currentMinFaceDetectionConfidence,
                minFaceTrackingConfidence = viewModel.currentMinFaceTrackingConfidence,
                minFacePresenceConfidence = viewModel.currentMinFacePresenceConfidence,
                maxNumFaces = viewModel.currentMaxFaces,
                currentDelegate = viewModel.currentDelegate,
                faceLandmarkerHelperListener = this
            )
        }

        initBottomSheetControls()

        fragmentCameraBinding.cameraCaptureButton.setOnClickListener {
            if (!isAutomatedCaptureRunning) {
                startAutomatedCapture()
            }
        }

        fragmentCameraBinding.stopButton.setOnClickListener {
            stopAutomatedCapture()
        }

        fragmentCameraBinding.cameraSwitchButton.setBackgroundColor(Color.BLUE)
        fragmentCameraBinding.cameraSwitchButton.setOnClickListener {
            switchCamera()
        }

        fragmentCameraBinding.flashButton.setBackgroundColor(Color.BLUE)
        fragmentCameraBinding.flashButton.setOnClickListener {
            toggleFlash()
        }

        fragmentCameraBinding.btnRaw.setOnClickListener {
            isRawMode = !isRawMode
            if (isRawMode) {
                isBurstMode = false
                fragmentCameraBinding.btnBurst.setBackgroundColor(Color.TRANSPARENT)
                fragmentCameraBinding.btnRaw.setBackgroundColor(Color.BLUE)
            } else {
                fragmentCameraBinding.btnRaw.setBackgroundColor(Color.TRANSPARENT)
            }
            switchCameraInternal()
        }

        fragmentCameraBinding.btnBurst.setOnClickListener {
            isBurstMode = !isBurstMode
            if (isBurstMode) {
                isRawMode = false
                fragmentCameraBinding.btnRaw.setBackgroundColor(Color.TRANSPARENT)
                fragmentCameraBinding.btnBurst.setBackgroundColor(Color.BLUE)
            } else {
                fragmentCameraBinding.btnBurst.setBackgroundColor(Color.TRANSPARENT)
            }
        }

        fragmentCameraBinding.backButton.setOnClickListener {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(R.id.action_camera_to_login)
        }
    }


    private val surfaceTextureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            openCamera(width, height)
        }

        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {
            adjustAspectRatio(width, height)
        }

        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean = true
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("CameraBackground").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera(width: Int, height: Int) {
        cameraManager = requireContext().getSystemService(Context.CAMERA_SERVICE) as CameraManager
        logAllCameras()
        logLogicalBackPhysicals()
        setUpCameraOutputs(width, height)
        adjustAspectRatio(width, height)
        try {
            if (!PermissionsFragment.hasPermissions(requireContext())) return
            if (cameraId == null) {
                Log.e(TAG, "No suitable camera found")
                return
            }
            cameraManager.openCamera(cameraId!!, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    
                    createCameraPreviewSession()
                }
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                }
                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    cameraDevice = null
                }
            }, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to open camera", e)
        }
    }

    private fun logAllCameras() {
        try {
            for (id in cameraManager.cameraIdList) {
                val c = cameraManager.getCameraCharacteristics(id)
                val facing = c.get(CameraCharacteristics.LENS_FACING)
                val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.joinToString()
                val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.joinToString()
                val phys = if (android.os.Build.VERSION.SDK_INT >= 28) c.physicalCameraIds.joinToString() else "n/a"
                Log.d(TAG, "camId=$id facing=$facing focal=[$focal] caps=[$caps] physicalIds=[$phys]")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error logging cameras", e)
        }
    }

    private fun logLogicalBackPhysicals() {
        try {
            for (id in cameraManager.cameraIdList) {
                val c = cameraManager.getCameraCharacteristics(id)
                val facing = c.get(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue

                val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
                val isLogical = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)
                if (!isLogical || android.os.Build.VERSION.SDK_INT < 28) continue

                Log.d(TAG, "LOGICAL_BACK=$id physicalIds=${c.physicalCameraIds.joinToString()}")

                for (pid in c.physicalCameraIds) {
                    val pc = cameraManager.getCameraCharacteristics(pid)
                    val focal = pc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.joinToString()
                    val active = pc.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                    Log.d(TAG, "  PID=$pid focal=[$focal] active=$active")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "logLogicalBackPhysicals failed", e)
        }
    }

    private fun switchCamera() {
        isFrontCamera = !isFrontCamera
        currentCameraIndex = 0
        switchCameraInternal()
    }

    private fun switchCameraInternal() {
        closeCamera()
        if (fragmentCameraBinding.viewFinder.isAvailable) {
            openCamera(fragmentCameraBinding.viewFinder.width, fragmentCameraBinding.viewFinder.height)
        }
    }

    private fun setUpCameraOutputs(width: Int, height: Int) {
        try {
            cameraId = null
            currentCharacteristics = null

            val targetFacing = if (isFrontCamera)
                CameraCharacteristics.LENS_FACING_FRONT
            else
                CameraCharacteristics.LENS_FACING_BACK

            availableLenses = queryLensesForFacing(targetFacing)
            
            availableLenses.forEachIndexed { i, l ->
                Log.d(TAG, "LENS[$i] open=${l.openCameraId} phys=${l.physicalId} focal=${l.focalMm} area=${l.sensorArea} label=${l.label}")
            }

            if (availableLenses.isEmpty()) {
                Log.e(TAG, "No cameras found for facing: $targetFacing")
                return
            }

            var bestIndex = 0

            if (isFrontCamera) {
                // Front: Prefer Narrowest FOV (largest focal length) to avoid ultra-wide
                bestIndex = pickFrontNormalIndex()
            } else {
                 val id0Index = availableLenses.indexOfFirst { it.openCameraId == "0" && it.physicalId == null }
                 if (id0Index >= 0) {
                     bestIndex = id0Index
                 } else {
                     val maxAreaIndex = availableLenses.indices.maxByOrNull { availableLenses[it].sensorArea }
                     if (maxAreaIndex != null) bestIndex = maxAreaIndex
                 }
            }
            
            Log.d(TAG, "CHOSEN lens index=$bestIndex => ${availableLenses[bestIndex].label}")

            currentCameraIndex = bestIndex
            val opt = availableLenses[currentCameraIndex]
            cameraId = opt.openCameraId
            selectedPhysicalId = opt.physicalId

            val characteristicsId = selectedPhysicalId ?: cameraId!!
            currentCharacteristics = cameraManager.getCameraCharacteristics(characteristicsId)

            if (cameraId != null && currentCharacteristics != null) {
                val characteristics = currentCharacteristics!!
                val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)

                zoomRatioRange = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
                    characteristics.get(CameraCharacteristics.CONTROL_ZOOM_RATIO_RANGE)
                } else null

                maxZoom = zoomRatioRange?.upper ?: characteristics.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
                activeArraySize = characteristics.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0

                if (map != null) {
                    if (isRawMode) {
                        val rawSizes = map.getOutputSizes(ImageFormat.RAW_SENSOR)
                        Log.d(TAG, "RAW sizes for camId=$cameraId: ${rawSizes?.joinToString()}")
                    }

                    previewSize = chooseOptimalSize(map.getOutputSizes(SurfaceTexture::class.java),
                        width, height)

                    val largest = map.getOutputSizes(ImageFormat.JPEG)?.maxByOrNull { it.width * it.height }
                    if (largest != null) {
                        imageReader = ImageReader.newInstance(largest.width, largest.height, ImageFormat.JPEG, 10)
                        imageReader?.setOnImageAvailableListener({ reader ->
                            backgroundExecutor.execute {
                                val image = reader.acquireNextImage() ?: return@execute
                                val timestamp = image.timestamp
                                val buffer = image.planes[0].buffer
                                val bytes = ByteArray(buffer.remaining())
                                buffer.get(bytes)
                                image.close()

                                val filename = filenameMap.remove(timestamp)
                                if (filename != null) {
                                    saveImage(bytes, filename)
                                } else {
                                    saveImage(bytes, "pic_${System.currentTimeMillis()}.jpg")
                                }
                            }
                        }, backgroundHandler)
                    }

                    if (isRawMode) {
                        val rawSizes = map.getOutputSizes(ImageFormat.RAW_SENSOR)
                        if (rawSizes != null && rawSizes.isNotEmpty()) {
                            val largestRaw = rawSizes.maxByOrNull { it.width * it.height }!!
                            rawImageReader = ImageReader.newInstance(largestRaw.width, largestRaw.height, ImageFormat.RAW_SENSOR, 10)
                            rawImageReader?.setOnImageAvailableListener({ reader ->
                                backgroundExecutor.execute {
                                    val image = reader.acquireNextImage() ?: return@execute
                                    synchronized(rawImageQueue) {
                                        rawImageQueue[image.timestamp] = image
                                        checkAndSaveMatchedRaw()
                                    }
                                }
                            }, backgroundHandler)
                        } else {
                            activity?.runOnUiThread {
                                Toast.makeText(requireContext(), "RAW not supported on this lens", Toast.LENGTH_SHORT).show()
                            }
                            isRawMode = false
                        }
                    }
                }
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun pickFrontNormalIndex(): Int {
        if (availableLenses.isEmpty()) return 0

        fun hfovRad(opt: LensOption): Double {
            val id = opt.physicalId ?: opt.openCameraId
            val c = try {
                cameraManager.getCameraCharacteristics(id)
            } catch (e: Exception) {
                return Double.POSITIVE_INFINITY
            }

            val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                ?.minOrNull()?.toDouble() ?: return Double.POSITIVE_INFINITY

            val phys = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                ?: return Double.POSITIVE_INFINITY

            return 2.0 * atan((phys.width.toDouble() / 2.0) / focal)
        }

        val indices = availableLenses.indices.toList()
        val physIdx = indices.filter { availableLenses[it].physicalId != null }
        val pool = if (physIdx.isNotEmpty()) physIdx else indices

        return pool.minByOrNull { hfovRad(availableLenses[it]) } ?: 0
    }

    private fun chooseOptimalSize(choices: Array<Size>, textureViewWidth: Int, textureViewHeight: Int): Size {
        return choices.sortedByDescending { it.width * it.height }.first()
    }

    private fun adjustAspectRatio(viewWidth: Int, viewHeight: Int) {
        if (previewSize == null) return

        val bufferWidth = previewSize!!.height
        val bufferHeight = previewSize!!.width

        activity?.runOnUiThread {
            val params = fragmentCameraBinding.viewFinder.layoutParams
            val overlayParams = fragmentCameraBinding.overlay.layoutParams

            val ratio = bufferWidth.toFloat() / bufferHeight.toFloat()

            if (viewWidth < viewHeight * ratio) {
                params.height = viewHeight
                params.width = (viewHeight * ratio).toInt()
            } else {
                params.width = viewWidth
                params.height = (viewWidth / ratio).toInt()
            }

            fragmentCameraBinding.viewFinder.layoutParams = params
            overlayParams.width = params.width
            overlayParams.height = params.height
            fragmentCameraBinding.overlay.layoutParams = overlayParams

            if (isFrontCamera) {
                val matrix = Matrix()
                matrix.setScale(1f, 1f, params.width / 2f, params.height / 2f)
                fragmentCameraBinding.viewFinder.setTransform(matrix)
            } else {
                fragmentCameraBinding.viewFinder.setTransform(null)
            }
        }
    }

    private fun createCameraPreviewSession() {
        try {
            val texture = fragmentCameraBinding.viewFinder.surfaceTexture!!

            if (previewSize != null) {
                texture.setDefaultBufferSize(previewSize!!.width, previewSize!!.height)
            } else {
                texture.setDefaultBufferSize(fragmentCameraBinding.viewFinder.width, fragmentCameraBinding.viewFinder.height)
            }

            val surface = Surface(texture)

            val usePhysical = (android.os.Build.VERSION.SDK_INT >= 28 && selectedPhysicalId != null)

            if (usePhysical) {
                val pid = selectedPhysicalId!!
                val outConfigs = mutableListOf<OutputConfiguration>()

                fun cfg(s: Surface) = OutputConfiguration(s).apply { setPhysicalCameraId(pid) }

                outConfigs.add(cfg(surface))
                imageReader?.surface?.let { outConfigs.add(cfg(it)) }
                rawImageReader?.surface?.let { outConfigs.add(cfg(it)) }

                val sessionConfig = SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    outConfigs,
                    Executors.newSingleThreadExecutor(),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(session: CameraCaptureSession) {
                            if (cameraDevice == null) return
                            captureSession = session
                            try {
                                previewRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                                previewRequestBuilder!!.addTarget(surface)
                                
                                previewRequestBuilder!!.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                                val startZoom = if (isFrontCamera) FRONT_DEFAULT_ZOOM else ZOOM_LEVEL_WIDE
                                setZoom(startZoom)
                                startAnalysis()
                                session.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
                            } catch (e: CameraAccessException) { e.printStackTrace() }
                        }
                        override fun onConfigureFailed(session: CameraCaptureSession) {}
                    }
                )
                cameraDevice!!.createCaptureSession(sessionConfig)
            } else {
                val targets = mutableListOf(surface)
                imageReader?.surface?.let { targets.add(it) }
                rawImageReader?.surface?.let { targets.add(it) }

                previewRequestBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                previewRequestBuilder!!.addTarget(surface)

                cameraDevice!!.createCaptureSession(targets, object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        captureSession = session
                        try {
                            previewRequestBuilder!!.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                            val startZoom = if (isFrontCamera) FRONT_DEFAULT_ZOOM else ZOOM_LEVEL_WIDE
                            setZoom(startZoom)
                            startAnalysis()
                            session.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
                        } catch (e: CameraAccessException) { e.printStackTrace() }
                    }
                    override fun onConfigureFailed(session: CameraCaptureSession) {}
                }, null)
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun closeCamera() {
        stopAnalysis()
        captureSession?.close()
        captureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
        rawImageReader?.close()
        rawImageReader = null
    }

    private fun startAnalysis() {
        isAnalyzing = true
        backgroundExecutor.execute(analysisRunnable)
    }

    private fun stopAnalysis() {
        isAnalyzing = false
    }

    private val analysisRunnable = object : Runnable {
        override fun run() {
            if (!isAnalyzing || _fragmentCameraBinding == null) return

            val viewFinder = fragmentCameraBinding.viewFinder
            if (viewFinder.isAvailable) {
                if (analysisBitmap == null || analysisBitmap!!.width != viewFinder.width || analysisBitmap!!.height != viewFinder.height) {
                    analysisBitmap = Bitmap.createBitmap(viewFinder.width, viewFinder.height, Bitmap.Config.ARGB_8888)
                }
                viewFinder.getBitmap(analysisBitmap!!)

                analysisBitmap?.let {
                    faceLandmarkerHelper.detectLiveStream(it, false, 0)
                }
            }

            if (isAnalyzing && !backgroundExecutor.isShutdown) {
                try {
                    Thread.sleep(30)
                    backgroundExecutor.execute(this)
                } catch (e: Exception) {

                }
            }
        }
    }

    private fun toggleFlash() {
        isFlashOn = !isFlashOn
        try {
            if (isFlashOn) {
                previewRequestBuilder?.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_TORCH)
            } else {
                previewRequestBuilder?.set(CaptureRequest.FLASH_MODE, CaptureRequest.FLASH_MODE_OFF)
            }
            captureSession?.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
        } catch (e: Exception) { e.printStackTrace() }
    }

    private fun setZoom(zoomLevel: Float, centerX: Float? = null, centerY: Float? = null) {
        val session = captureSession ?: return
        val builder = previewRequestBuilder ?: return

        try {
            val z = zoomLevel.coerceIn(1f, maxZoom)

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && zoomRatioRange != null) {
                builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, z)

                if (centerX != null && centerY != null && activeArraySize != null) {
                    val aa = activeArraySize!!
                    val cropW = (aa.width() / z).toInt()
                    val cropH = (aa.height() / z).toInt()
                    val cropX = (centerX - cropW / 2f).toInt().coerceIn(aa.left, aa.right - cropW)
                    val cropY = (centerY - cropH / 2f).toInt().coerceIn(aa.top, aa.bottom - cropH)
                    builder.set(CaptureRequest.SCALER_CROP_REGION, Rect(cropX, cropY, cropX + cropW, cropY + cropH))
                } else {
                    builder.set(CaptureRequest.SCALER_CROP_REGION, null)
                }
            } else {
                val aa = activeArraySize ?: return
                val cropW = (aa.width() / z).toInt()
                val cropH = (aa.height() / z).toInt()

                val cx = (centerX ?: aa.exactCenterX().toFloat())
                val cy = (centerY ?: aa.exactCenterY().toFloat())

                val cropX = (cx - cropW / 2f).toInt().coerceIn(aa.left, aa.right - cropW)
                val cropY = (cy - cropH / 2f).toInt().coerceIn(aa.top, aa.bottom - cropH)

                builder.set(CaptureRequest.SCALER_CROP_REGION, Rect(cropX, cropY, cropX + cropW, cropY + cropH))
            }

            session.setRepeatingRequest(builder.build(), repeatingPreviewCallback, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "setZoom failed", e)
        }
    }

    private fun unlock3AAndZoomOutTo1x() {
        val session = captureSession ?: return
        val builder = previewRequestBuilder ?: return

        try {
            builder.set(CaptureRequest.CONTROL_AWB_LOCK, false)
            builder.set(CaptureRequest.CONTROL_AE_LOCK, false)
            builder.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL)
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)

            builder.set(CaptureRequest.CONTROL_AF_REGIONS, null)
            builder.set(CaptureRequest.CONTROL_AE_REGIONS, null)

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && zoomRatioRange != null) {
                builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, 1.0f)
                activeArraySize?.let { builder.set(CaptureRequest.SCALER_CROP_REGION, it) }
            } else {
                activeArraySize?.let { builder.set(CaptureRequest.SCALER_CROP_REGION, it) }
            }

            session.setRepeatingRequest(builder.build(), repeatingPreviewCallback, backgroundHandler)
        } catch (e: Exception) {
            Log.e(TAG, "unlock3AAndZoomOutTo1x failed", e)
        }
    }

    private fun showStatus(message: String) {
        val binding = _fragmentCameraBinding ?: return
        activity?.runOnUiThread {
            binding.statusText.visibility = View.VISIBLE
            binding.statusText.text = message
        }
    }

    private fun startAutomatedCapture() {
        if (isAutomatedCaptureRunning) return
        isAutomatedCaptureRunning = true

        setZoom(ZOOM_LEVEL_WIDE)

        fragmentCameraBinding.overlay.frozenEyeBox = null
        fragmentCameraBinding.overlay.eyeAlignmentBox = null

        fragmentCameraBinding.cameraCaptureButton.visibility = View.GONE
        fragmentCameraBinding.stopButton.visibility = View.VISIBLE
        fragmentCameraBinding.cameraSwitchButton.isEnabled = false

        captureJob = lifecycleScope.launch(Dispatchers.IO) {
            try {
                performEyeCapture(true)

                unlock3AAndZoomOutTo1x()
                delay(1500)

                performEyeCapture(false)
            } catch (e: Exception) {
                Log.d(TAG, "Capture interrupted: ${e.message}")
            } finally {
                withContext(Dispatchers.Main) {
                    resetAfterCapture()
                }
            }
        }
    }

    private fun stopAutomatedCapture() {
        captureJob?.cancel()
        captureJob = null
        isAutomatedCaptureRunning = false

        if (Thread.currentThread() == android.os.Looper.getMainLooper().thread) {
            resetAfterCapture()
        } else {
            activity?.runOnUiThread { resetAfterCapture() }
        }
    }

    private fun resetAfterCapture() {
        if (_fragmentCameraBinding == null) return

        fragmentCameraBinding.overlay.showEyeAlignmentBox = false
        fragmentCameraBinding.overlay.frozenEyeBox = null
        fragmentCameraBinding.overlay.isTargetingRightEye = null
        fragmentCameraBinding.overlay.eyeAlignmentBox = null
        fragmentCameraBinding.overlay.focusRing = null

        fragmentCameraBinding.cameraCaptureButton.visibility = View.VISIBLE
        fragmentCameraBinding.stopButton.visibility = View.GONE
        fragmentCameraBinding.cameraSwitchButton.isEnabled = true
        fragmentCameraBinding.statusText.visibility = View.GONE

        backgroundHandler?.post {
            try {
                captureSession?.stopRepeating()
                captureSession?.abortCaptures()
            } catch (_: Exception) { }

            unlock3AAndZoomOutTo1x()
        }

        fragmentCameraBinding.overlay.invalidate()
        isAutomatedCaptureRunning = false
    }

    private fun smoothRect(newR: RectF, a: Float = 0.25f): RectF {
        val prev = smoothBox
        val out = if (prev == null) newR else RectF(
            prev.left + a * (newR.left - prev.left),
            prev.top + a * (newR.top - prev.top),
            prev.right + a * (newR.right - prev.right),
            prev.bottom + a * (newR.bottom - prev.bottom)
        )
        smoothBox = out
        return out
    }

    private fun meteringFromEyeBox(
        eyeBoxPx: RectF,
        imgW: Int,
        imgH: Int
    ): MeteringRectangle? {
        val crop = lastCropRegion ?: activeArraySize ?: return null

        var nx0 = (eyeBoxPx.left / imgW).coerceIn(0f, 1f)
        var ny0 = (eyeBoxPx.top / imgH).coerceIn(0f, 1f)
        var nx1 = (eyeBoxPx.right / imgW).coerceIn(0f, 1f)
        var ny1 = (eyeBoxPx.bottom / imgH).coerceIn(0f, 1f)

        val cx = (nx0 + nx1) / 2f
        val cy = (ny0 + ny1) / 2f
        val minFrac = 0.25f
        val halfW = max((nx1 - nx0) / 2f, minFrac / 2f)
        val halfH = max((ny1 - ny0) / 2f, minFrac / 2f)

        nx0 = (cx - halfW).coerceIn(0f, 1f)
        nx1 = (cx + halfW).coerceIn(0f, 1f)
        ny0 = (cy - halfH).coerceIn(0f, 1f)
        ny1 = (cy + halfH).coerceIn(0f, 1f)

        if (isFrontCamera) {
            val mx0 = 1f - nx1
            val mx1 = 1f - nx0
            nx0 = mx0
            nx1 = mx1
        }

        fun map(nx: Float, ny: Float): PointF {
            val w = crop.width().toFloat()
            val h = crop.height().toFloat()

            return when (sensorOrientation) {
                90 -> PointF(
                    crop.left + ny * w,
                    crop.top + (1f - nx) * h
                )
                270 -> PointF(
                    crop.left + (1f - ny) * w,
                    crop.top + nx * h
                )
                else -> PointF(
                    crop.left + nx * w,
                    crop.top + ny * h
                )
            }
        }

        val p00 = map(nx0, ny0)
        val p11 = map(nx1, ny1)
        val left = min(p00.x, p11.x).toInt()
        val top = min(p00.y, p11.y).toInt()
        val right = max(p00.x, p11.x).toInt()
        val bottom = max(p00.y, p11.y).toInt()

        val r = Rect(left, top, right, bottom)
        r.intersect(activeArraySize!!)
        if (r.width() <= 0 || r.height() <= 0) return null

        return MeteringRectangle(r, MeteringRectangle.METERING_WEIGHT_MAX)
    }

    private suspend fun performEyeCapture(isRightEye: Boolean) {
        withContext(Dispatchers.Main) {
            fragmentCameraBinding.overlay.isTargetingRightEye = isRightEye
            fragmentCameraBinding.overlay.frozenEyeBox = null
            fragmentCameraBinding.overlay.eyeAlignmentBox = null
            fragmentCameraBinding.overlay.showEyeAlignmentBox = true
            showStatus(if (isRightEye) "Align Right Eye" else "Align Left Eye")
        }

        var openRetries = 0
        while (openRetries < 50) {
            val openness = if (isRightEye) rightEyeOpenness else leftEyeOpenness
            if (openness > 0.5f) break
            showStatus(if (isRightEye) "Open Right Eye" else "Open Left Eye")
            delay(100)
            openRetries++
        }

        delay(1500)

        var targetBox: RectF? = null
        withContext(Dispatchers.Main) {
            targetBox = fragmentCameraBinding.overlay.eyeAlignmentBox
            if (targetBox != null) {
                // Freeze visual so user knows what we are targeting
                fragmentCameraBinding.overlay.frozenEyeBox = targetBox
                fragmentCameraBinding.overlay.invalidate()
            }
        }

        if (targetBox == null) {
            showStatus("Eye not found, retrying...")
            return
        }

        // Use the center of the target box to center the zoom
        var sensorCx = 0f
        var sensorCy = 0f
        
        // Determine capture zoom level
        // If back camera and telephoto lens, avoid digital zoom (stay at 1x)
        var captureZoom = ZOOM_LEVEL_CAPTURE
        if (!isFrontCamera && availableLenses.isNotEmpty() && currentCameraIndex < availableLenses.size) {
             val focal = availableLenses[currentCameraIndex].focalMm
             if (focal != null && focal > 5.0f) {
                 captureZoom = 1.0f
             }
        }

        if (activeArraySize != null && mpInputImageWidth > 0 && mpInputImageHeight > 0) {
            val cx = targetBox!!.centerX()
            val cy = targetBox!!.centerY()

            var nx = cx / mpInputImageWidth
            var ny = cy / mpInputImageHeight

            if (isFrontCamera) nx = 1.0f - nx

            when (sensorOrientation) {
                270 -> {
                    sensorCx = (1f - ny) * activeArraySize!!.width() + activeArraySize!!.left
                    sensorCy = nx * activeArraySize!!.height() + activeArraySize!!.top
                }
                90 -> {
                    sensorCx = ny * activeArraySize!!.width() + activeArraySize!!.left
                    sensorCy = (1f - nx) * activeArraySize!!.height() + activeArraySize!!.top
                }
                else -> {
                    sensorCx = nx * activeArraySize!!.width() + activeArraySize!!.left
                    sensorCy = ny * activeArraySize!!.height() + activeArraySize!!.top
                }
            }

            val aa = activeArraySize!!
            val z = captureZoom.coerceIn(1f, maxZoom)
            val cropW = (aa.width() / z).toInt()
            val cropH = (aa.height() / z).toInt()
            val cropX = (sensorCx - cropW / 2f).toInt().coerceIn(aa.left, aa.right - cropW)
            val cropY = (sensorCy - cropH / 2f).toInt().coerceIn(aa.top, aa.bottom - cropH)

            val sensorCxRel = sensorCx - cropX
            val sensorCyRel = sensorCy - cropY

            var nxNew = 0f
            var nyNew = 0f

            when (sensorOrientation) {
                270 -> {
                    nyNew = 1f - (sensorCxRel / cropW)
                    nxNew = sensorCyRel / cropH
                }
                90 -> {
                    nyNew = sensorCxRel / cropW
                    nxNew = 1f - (sensorCyRel / cropH)
                }
                else -> {
                    nxNew = sensorCxRel / cropW
                    nyNew = sensorCyRel / cropH
                }
            }

            if (isFrontCamera) nxNew = 1f - nxNew

            val newCx = nxNew * mpInputImageWidth
            val newCy = nyNew * mpInputImageHeight
            val newW = targetBox!!.width() * z
            val newH = targetBox!!.height() * z

            val newBox = RectF(
                newCx - newW / 2,
                newCy - newH / 2,
                newCx + newW / 2,
                newCy + newH / 2
            )

            withContext(Dispatchers.Main) {
                fragmentCameraBinding.overlay.frozenEyeBox = newBox
                fragmentCameraBinding.overlay.invalidate()
            }
        } else {
            activeArraySize?.let {
                sensorCx = it.centerX().toFloat()
                sensorCy = it.centerY().toFloat()
            }
        }

        setZoom(captureZoom, sensorCx, sensorCy)

        delay(3000)

        // FOCUS & EXPOSURE LOCK

        var meteringBox: RectF? = null
        val currentTargetBox = fragmentCameraBinding.overlay.frozenEyeBox
        
        if (currentTargetBox != null) {
            val w = currentTargetBox.width()
            val h = currentTargetBox.height()
            val cx = currentTargetBox.centerX()
            val cy = currentTargetBox.centerY()
            val expandM = 2.0f
            meteringBox = RectF(
                cx - (w * expandM)/2,
                cy - (h * expandM)/2,
                cx + (w * expandM)/2,
                cy + (h * expandM)/2
            )
            
            withContext(Dispatchers.Main) {
                fragmentCameraBinding.overlay.focusRing = meteringBox
                fragmentCameraBinding.overlay.invalidate()
            }
        }
        
        if (meteringBox != null) {
            val metering = meteringFromEyeBox(meteringBox, mpInputImageWidth, mpInputImageHeight)

            if (metering != null) {
                try {
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AF_REGIONS, arrayOf(metering))
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AE_REGIONS, arrayOf(metering))
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER, CaptureRequest.CONTROL_AE_PRECAPTURE_TRIGGER_START)
                    captureSession!!.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
                    delay(800)
    
                    triggerFocus()
    
                    // LOCK Everything (IDLE trigger, AE/AWB Lock)
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_IDLE)
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AE_LOCK, true)
                    previewRequestBuilder?.set(CaptureRequest.CONTROL_AWB_LOCK, true)
                    captureSession!!.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
                    delay(300)
                } catch (e: Exception) { e.printStackTrace() }
            }
        }


        for (w in 0..4) {
            takePicture("Warmup", w, "X")
            delay(150)
        }

        val count = if (isBurstMode) 10 else 5
        val mode = if (isRawMode) "R" else if (isBurstMode) "B" else "N"
        val eyeLabel = if (isRightEye) "Right" else "Left"

        for (i in 1..count) {
            takePicture(eyeLabel, i, mode)
            if (isBurstMode) delay(100) else delay(1000)
        }

        delay(500)

        try {
            captureSession?.stopRepeating()
        } catch (_: Exception) { }

        unlock3AAndZoomOutTo1x()
        delay(400)

        withContext(Dispatchers.Main) {
            fragmentCameraBinding.overlay.frozenEyeBox = null
            fragmentCameraBinding.overlay.showEyeAlignmentBox = false
            fragmentCameraBinding.overlay.focusRing = null
            fragmentCameraBinding.overlay.invalidate()
        }
    }

    private suspend fun triggerFocus(): Boolean = suspendCancellableCoroutine { cont ->
        try {
            previewRequestBuilder?.set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START)

            val callback = object : CameraCaptureSession.CaptureCallback() {
                private var isResumed = false

                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    lastCropRegion = result.get(CaptureResult.SCALER_CROP_REGION)

                    val afState = result.get(CaptureResult.CONTROL_AF_STATE)
                    if (afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                        afState == CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED) {
                        if (!isResumed) {
                            isResumed = true
                            if (cont.isActive) {
                                cont.resume(afState == CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED)
                            }
                            try {
                                previewRequestBuilder?.set(CaptureRequest.CONTROL_AF_TRIGGER, null)
                                session.setRepeatingRequest(previewRequestBuilder!!.build(), repeatingPreviewCallback, backgroundHandler)
                            } catch (e: Exception) { }
                        }
                    }
                }
            }
            captureSession?.setRepeatingRequest(previewRequestBuilder!!.build(), callback, backgroundHandler)
        } catch (e: Exception) {
            if (cont.isActive) cont.resume(false)
        }
    }

    private fun takePicture(eye: String, number: Int, mode: String) {
        if (cameraDevice == null || captureSession == null) return
        try {
            val captureBuilder = cameraDevice!!.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)

            imageReader?.surface?.let { captureBuilder.addTarget(it) }

            if (isRawMode && rawImageReader != null) {
                captureBuilder.addTarget(rawImageReader!!.surface)
            }

            applyCommonCaptureSettings(captureBuilder)

            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && zoomRatioRange != null) {
                val z = (previewRequestBuilder?.get(CaptureRequest.CONTROL_ZOOM_RATIO) ?: ZOOM_LEVEL_WIDE)
                    .coerceIn(zoomRatioRange!!.lower, zoomRatioRange!!.upper)

                captureBuilder.set(CaptureRequest.CONTROL_ZOOM_RATIO, z)

                previewRequestBuilder?.get(CaptureRequest.SCALER_CROP_REGION)?.let {
                    captureBuilder.set(CaptureRequest.SCALER_CROP_REGION, it)
                }
            } else {
                previewRequestBuilder?.get(CaptureRequest.SCALER_CROP_REGION)?.let {
                    captureBuilder.set(CaptureRequest.SCALER_CROP_REGION, it)
                }
            }

            val rotation = requireActivity().windowManager.defaultDisplay.rotation
            captureBuilder.set(CaptureRequest.JPEG_ORIENTATION, getOrientation(rotation))

            val cameraLabel = if (isFrontCamera) "Front" else "Back"

            val filename = "${viewModel.participantId}_${cameraLabel}_${eye}_${number}_${mode}"
            captureBuilder.setTag(filename)

            val captureCallback = object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureStarted(session: CameraCaptureSession, request: CaptureRequest, timestamp: Long, frameNumber: Long) {
                    super.onCaptureStarted(session, request, timestamp, frameNumber)
                    val tag = request.tag as? String
                    if (tag != null) {
                        filenameMap[timestamp] = tag
                    }
                }

                override fun onCaptureCompleted(
                    session: CameraCaptureSession,
                    request: CaptureRequest,
                    result: TotalCaptureResult
                ) {
                    if (isRawMode) {
                        synchronized(rawResultQueue) {
                            rawResultQueue[result.get(CaptureResult.SENSOR_TIMESTAMP)!!] = result
                            checkAndSaveMatchedRaw()
                        }
                    }
                }
            }

            captureSession?.capture(captureBuilder.build(), captureCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun takePicture() {
        takePicture("Manual", 0, "N")
    }

    private fun checkAndSaveMatchedRaw() {
        val matchedTimestamps = rawResultQueue.keys.intersect(rawImageQueue.keys)
        for (ts in matchedTimestamps) {
            val result = rawResultQueue.remove(ts)!!
            val image = rawImageQueue.remove(ts)!!

            // Check if warmup
            val tag = result.request.tag as? String
            if (tag != null && tag.contains("Warmup")) {
                image.close()
                continue
            }

            val characteristics = currentCharacteristics ?: continue
            val filename = tag ?: "pic_${ts}"
            saveRawImage(image, characteristics, result, filename)
            image.close()
        }
    }

    private fun takeBurstPicture(count: Int) {}

    private fun applyCommonCaptureSettings(builder: CaptureRequest.Builder) {
        builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
        builder.set(CaptureRequest.JPEG_QUALITY, 100.toByte())
        builder.set(CaptureRequest.EDGE_MODE, CaptureRequest.EDGE_MODE_HIGH_QUALITY)
        builder.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY)

        previewRequestBuilder?.get(CaptureRequest.SCALER_CROP_REGION)?.let {
            builder.set(CaptureRequest.SCALER_CROP_REGION, it)
        }

        // Copy Locks
        previewRequestBuilder?.get(CaptureRequest.CONTROL_AWB_LOCK)?.let {
            builder.set(CaptureRequest.CONTROL_AWB_LOCK, it)
        }
        previewRequestBuilder?.get(CaptureRequest.CONTROL_AE_LOCK)?.let {
            builder.set(CaptureRequest.CONTROL_AE_LOCK, it)
        }

        previewRequestBuilder?.get(CaptureRequest.CONTROL_AF_REGIONS)?.let {
            builder.set(CaptureRequest.CONTROL_AF_REGIONS, it)
        }
        previewRequestBuilder?.get(CaptureRequest.CONTROL_AE_REGIONS)?.let {
            builder.set(CaptureRequest.CONTROL_AE_REGIONS, it)
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R && zoomRatioRange != null) {
            previewRequestBuilder?.get(CaptureRequest.CONTROL_ZOOM_RATIO)?.let {
                builder.set(CaptureRequest.CONTROL_ZOOM_RATIO, it)
            }
        }
    }

    private fun getOrientation(rotation: Int): Int {
        if (cameraId == null) return 0
        val sensorOrientation = try {
            cameraManager.getCameraCharacteristics(cameraId!!).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 270
        } catch (e: Exception) {
            270
        }

        var deviceRotation = 0
        when(rotation) {
            Surface.ROTATION_0 -> deviceRotation = 0
            Surface.ROTATION_90 -> deviceRotation = 90
            Surface.ROTATION_180 -> deviceRotation = 180
            Surface.ROTATION_270 -> deviceRotation = 270
        }

        return (sensorOrientation + deviceRotation) % 360
    }

    private fun getExifOrientation(rotation: Int): Int {
        val degrees = getOrientation(rotation)
        return when (degrees) {
            0 -> ExifInterface.ORIENTATION_NORMAL
            90 -> ExifInterface.ORIENTATION_ROTATE_90
            180 -> ExifInterface.ORIENTATION_ROTATE_180
            270 -> ExifInterface.ORIENTATION_ROTATE_270
            else -> ExifInterface.ORIENTATION_NORMAL
        }
    }

    private fun saveImage(bytes: ByteArray, filenameBase: String) {
        if (filenameBase.contains("Warmup")) return

        val fileName = if (filenameBase.endsWith(".jpg")) filenameBase else "${filenameBase}.jpg"
        saveToMediaStore(bytes, fileName, "image/jpeg", "Pictures/FaceLandmarker")
    }

    private fun saveRawImage(image: Image, characteristics: CameraCharacteristics, result: TotalCaptureResult, filenameBase: String) {
        val fileName = if (filenameBase.endsWith(".dng")) filenameBase else "${filenameBase}.dng"

        val contentValues = android.content.ContentValues().apply {
            put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/x-adobe-dng")
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/FaceLandmarker")
                put(android.provider.MediaStore.MediaColumns.IS_PENDING, 1)
            }
        }

        val resolver = requireContext().contentResolver
        val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

        if (uri != null) {
            try {
                resolver.openOutputStream(uri)?.use { outputStream ->
                    DngCreator(characteristics, result).use { dngCreator ->
                        val rotation = requireActivity().windowManager.defaultDisplay.rotation
                        dngCreator.setOrientation(getExifOrientation(rotation))
                        dngCreator.writeImage(outputStream, image)
                    }
                }

                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(android.provider.MediaStore.MediaColumns.IS_PENDING, 0)
                    resolver.update(uri, contentValues, null, null)
                }

                activity?.runOnUiThread {
                    Toast.makeText(requireContext(), "Saved $fileName", Toast.LENGTH_SHORT).show()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun saveToMediaStore(bytes: ByteArray, fileName: String, mimeType: String, path: String) {
        var outputStream: java.io.OutputStream? = null
        try {
            val contentValues = android.content.ContentValues().apply {
                put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                put(android.provider.MediaStore.MediaColumns.MIME_TYPE, mimeType)
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    put(android.provider.MediaStore.MediaColumns.RELATIVE_PATH, path)
                    put(android.provider.MediaStore.MediaColumns.IS_PENDING, 1)
                }
            }

            val resolver = requireContext().contentResolver
            val uri = resolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)

            if (uri != null) {
                outputStream = resolver.openOutputStream(uri)
                outputStream?.write(bytes)
                outputStream?.flush()

                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(android.provider.MediaStore.MediaColumns.IS_PENDING, 0)
                    resolver.update(uri, contentValues, null, null)
                }

                requireActivity().runOnUiThread {
                    Toast.makeText(requireContext(), "Saved $fileName", Toast.LENGTH_SHORT).show()
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            outputStream?.close()
        }
    }

    // Laplacian Variance for Sharpness
    private fun calculateSharpness(bitmap: Bitmap?): Double {
        if (bitmap == null) return 0.0

        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // Convert to Grayscale
        val grayPixels = IntArray(width * height)
        for (i in pixels.indices) {
            val p = pixels[i]
            val r = (p shr 16) and 0xff
            val g = (p shr 8) and 0xff
            val b = p and 0xff
            // Luminance
            grayPixels[i] = (0.299 * r + 0.587 * g + 0.114 * b).toInt()
        }

        // Laplacian Kernel

        var sum = 0.0
        var sumSq = 0.0
        var count = 0

        // Convolve
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val center = grayPixels[y * width + x]
                val up = grayPixels[(y - 1) * width + x]
                val down = grayPixels[(y + 1) * width + x]
                val left = grayPixels[y * width + (x - 1)]
                val right = grayPixels[y * width + (x + 1)]

                val lap = up + down + left + right - 4 * center

                sum += lap
                sumSq += (lap * lap)
                count++
            }
        }

        val mean = sum / count
        val variance = (sumSq / count) - (mean * mean)

        return variance
    }

    private fun initBottomSheetControls() {
        fragmentCameraBinding.bottomSheetLayout.maxFacesValue.text =
            viewModel.currentMaxFaces.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFaceDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFaceTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US, "%.2f", viewModel.currentMinFacePresenceConfidence
            )

        // When clicked, lower face detection score threshold floor
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceDetectionConfidence >= 0.2) {
                faceLandmarkerHelper.minFaceDetectionConfidence -= 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.detectionThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceDetectionConfidence <= 0.8) {
                faceLandmarkerHelper.minFaceDetectionConfidence += 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.trackingThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceTrackingConfidence >= 0.2) {
                faceLandmarkerHelper.minFaceTrackingConfidence -= 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.trackingThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFaceTrackingConfidence <= 0.8) {
                faceLandmarkerHelper.minFaceTrackingConfidence += 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.presenceThresholdMinus.setOnClickListener {
            if (faceLandmarkerHelper.minFacePresenceConfidence >= 0.2) {
                faceLandmarkerHelper.minFacePresenceConfidence -= 0.1f
                updateControlsUi()
            }
        }

        fragmentCameraBinding.bottomSheetLayout.presenceThresholdPlus.setOnClickListener {
            if (faceLandmarkerHelper.minFacePresenceConfidence <= 0.8) {
                faceLandmarkerHelper.minFacePresenceConfidence += 0.1f
                updateControlsUi()
            }
        }


        fragmentCameraBinding.bottomSheetLayout.maxFacesMinus.setOnClickListener {
            if (faceLandmarkerHelper.maxNumFaces > 1) {
                faceLandmarkerHelper.maxNumFaces--
                updateControlsUi()
            }
        }


        fragmentCameraBinding.bottomSheetLayout.maxFacesPlus.setOnClickListener {
            if (faceLandmarkerHelper.maxNumFaces < 2) {
                faceLandmarkerHelper.maxNumFaces++
                updateControlsUi()
            }
        }

        // When clicked, change the underlying hardware used for inference.
        // Current options are CPU and GPU
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
            viewModel.currentDelegate, false
        )
        fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    p0: AdapterView<*>?, p1: View?, p2: Int, p3: Long
                ) {
                    try {
                        faceLandmarkerHelper.currentDelegate = p2
                        updateControlsUi()
                    } catch(e: UninitializedPropertyAccessException) {
                        Log.e(TAG, "FaceLandmarkerHelper has not been initialized yet.")
                    }
                }

                override fun onNothingSelected(p0: AdapterView<*>?) {
                    /* no op */
                }
            }
    }

    // Update the values displayed in the bottom sheet. Reset Facelandmarker
    // helper.
    private fun updateControlsUi() {
        fragmentCameraBinding.bottomSheetLayout.maxFacesValue.text =
            faceLandmarkerHelper.maxNumFaces.toString()
        fragmentCameraBinding.bottomSheetLayout.detectionThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFaceDetectionConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.trackingThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFaceTrackingConfidence
            )
        fragmentCameraBinding.bottomSheetLayout.presenceThresholdValue.text =
            String.format(
                Locale.US,
                "%.2f",
                faceLandmarkerHelper.minFacePresenceConfidence
            )

        // Needs to be cleared instead of reinitialized because the GPU
        // delegate needs to be initialized on the thread using it when applicable
        backgroundExecutor.execute {
            faceLandmarkerHelper.clearFaceLandmarker()
            faceLandmarkerHelper.setupFaceLandmarker()
        }
        fragmentCameraBinding.overlay.clear()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
    }

    // Update UI after face have been detected. Extracts original
    // image height/width to scale and place the landmarks properly through
    // OverlayView
    override fun onResults(
        resultBundle: FaceLandmarkerHelper.ResultBundle
    ) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                if (fragmentCameraBinding.recyclerviewResults.scrollState != SCROLL_STATE_DRAGGING) {
                    faceBlendshapesResultAdapter.updateResults(resultBundle.result)
                    faceBlendshapesResultAdapter.notifyDataSetChanged()
                }


                fragmentCameraBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", resultBundle.inferenceTime)

                // Pass necessary information to OverlayView for drawing on the canvas
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.result,
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

                // Store dims for metering
                mpInputImageWidth = resultBundle.inputImageWidth
                mpInputImageHeight = resultBundle.inputImageHeight

                if (isAutomatedCaptureRunning && resultBundle.result.faceLandmarks().isNotEmpty()) {
                    val landmarks = resultBundle.result.faceLandmarks()[0]

                    // Extract blendshapes for eye openness
                    if (resultBundle.result.faceBlendshapes().isPresent) {
                        val blendshapes = resultBundle.result.faceBlendshapes().get()[0]
                        blendshapes.forEach { category ->
                            when (category.categoryName()) {
                                "eyeBlinkLeft" -> leftEyeOpenness = 1.0f - category.score()
                                "eyeBlinkRight" -> rightEyeOpenness = 1.0f - category.score()
                            }
                        }
                    }

                    val isTargetingRight = fragmentCameraBinding.overlay.isTargetingRightEye
                    if (isTargetingRight != null) {
                        val indices = if (isTargetingRight == true) FaceLandmarkerHelper.RIGHT_EYE_LANDMARKS else FaceLandmarkerHelper.LEFT_EYE_LANDMARKS

                        var minX = 1f
                        var maxX = 0f
                        var minY = 1f
                        var maxY = 0f

                        indices.forEach { index ->
                            val landmark = landmarks[index]
                            minX = min(minX, landmark.x())
                            maxX = max(maxX, landmark.x())
                            minY = min(minY, landmark.y())
                            maxY = max(maxY, landmark.y())
                        }

                        val left = minX * resultBundle.inputImageWidth
                        val top = minY * resultBundle.inputImageHeight
                        val right = maxX * resultBundle.inputImageWidth
                        val bottom = maxY * resultBundle.inputImageHeight

                        val rawRect = RectF(left, top, right, bottom)

                        val width = rawRect.width()
                        val height = rawRect.height()
                        val cx = rawRect.centerX()
                        val cy = rawRect.centerY()

                        val expansion = 1.5f
                        val newHalfW = (width * expansion) / 2f
                        val newHalfH = (height * expansion) / 2f

                        val expandedRect = RectF(
                            max(0f, cx - newHalfW),
                            max(0f, cy - newHalfH),
                            min(resultBundle.inputImageWidth.toFloat(), cx + newHalfW),
                            min(resultBundle.inputImageHeight.toFloat(), cy + newHalfH)
                        )

                        fragmentCameraBinding.overlay.eyeAlignmentBox = smoothRect(expandedRect)
                    } else {
                        fragmentCameraBinding.overlay.eyeAlignmentBox = null
                    }
                } else if (!isAutomatedCaptureRunning) {
                    fragmentCameraBinding.overlay.eyeAlignmentBox = null
                }

                // Force a redraw
                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onEmpty() {
        fragmentCameraBinding.overlay.clear()
        activity?.runOnUiThread {
            faceBlendshapesResultAdapter.updateResults(null)
            faceBlendshapesResultAdapter.notifyDataSetChanged()
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
            faceBlendshapesResultAdapter.updateResults(null)
            faceBlendshapesResultAdapter.notifyDataSetChanged()

            if (errorCode == FaceLandmarkerHelper.GPU_ERROR) {
                fragmentCameraBinding.bottomSheetLayout.spinnerDelegate.setSelection(
                    FaceLandmarkerHelper.DELEGATE_CPU, false
                )
            }
        }
    }

    private fun queryLensesForFacing(facing: Int): List<LensOption> {
        val out = LinkedHashMap<String, LensOption>() // key = label/id combo

        fun sensorAreaOf(c: CameraCharacteristics): Int {
            val a = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
            return if (a != null) a.width() * a.height() else 0
        }

        fun addLogical(id: String) {
            val c = try { cameraManager.getCameraCharacteristics(id) } catch (_: Exception) { return }
            if (c.get(CameraCharacteristics.LENS_FACING) != facing) return

            val focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull()
            val map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val supportsRaw = map?.getOutputSizes(ImageFormat.RAW_SENSOR)?.isNotEmpty() == true
            
            val mm = if (focal != null) String.format(Locale.US, "%.1fmm", focal) else "?"
            val raw = if (supportsRaw) "RAW" else "no RAW"
            val area = sensorAreaOf(c)

            out["logical:$id"] = LensOption(
                openCameraId = id,
                physicalId = null,
                label = "Auto (logical) ($mm, $raw)",
                focalMm = focal,
                sensorArea = area
            )
        }
        
        fun addPhysical(logicalId: String, pid: String) {
            val pc = try { cameraManager.getCameraCharacteristics(pid) } catch (_: Exception) { return }
            if (pc.get(CameraCharacteristics.LENS_FACING) != facing) return

            val focal = pc.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.minOrNull()
            val map = pc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            val supportsRaw = map?.getOutputSizes(ImageFormat.RAW_SENSOR)?.isNotEmpty() == true

            val mm = if (focal != null) String.format(Locale.US, "%.1fmm", focal) else "?"
            val raw = if (supportsRaw) "RAW" else "no RAW"
            val area = sensorAreaOf(pc)

            out["phys:$logicalId:$pid"] = LensOption(
                openCameraId = logicalId,
                physicalId = pid,
                label = "Physical $pid ($mm, $raw)",
                focalMm = focal,
                sensorArea = area
            )
        }

        for (id in cameraManager.cameraIdList) {
            val c = try { cameraManager.getCameraCharacteristics(id) } catch (_: Exception) { continue }
            if (c.get(CameraCharacteristics.LENS_FACING) != facing) continue

            addLogical(id)
            
            val caps = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES) ?: intArrayOf()
            val isLogical = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_LOGICAL_MULTI_CAMERA)
             if (isLogical && android.os.Build.VERSION.SDK_INT >= 28) {
                for (pid in c.physicalCameraIds) {
                    addPhysical(id, pid)
                }
            }
        }

        return out.values.toList()
    }
}
