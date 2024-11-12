console.log('Iniciando servidor...');

const express = require('express');
const mongoose = require('mongoose'); 
const cors = require('cors');
require('dotenv').config();
const Parking = require('./models/parkings'); 
const Reservation = require('./models/reservations'); 
const User = require('./models/users')
const app = express();
const port = 4000;

mongoose.connect(process.env.MONGO_URI, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
    .then(() => console.log('Conexión exitosa a MongoDB Atlas con Mongoose'))
    .catch((error) => console.error('Error al conectarse a MongoDB con Mongoose:', error));

app.use(cors());
app.use(express.json());

app.get('/api/parkings', async (req, res) => {
    try {
        const parkings = await Parking.find(); 
        res.json(parkings);
    } catch (error) {
        res.status(500).json({ error: 'Error al obtener los estacionamientos' });
    }
});

app.post('/api/parkings', async (req, res) => {
    try {
        const { location, occupied, available, pricePerHour, currentUsers } = req.body;

        // Crear un nuevo documento utilizando el modelo Mongoose
        const newParking = new Parking({
            location,
            occupied: parseInt(occupied, 10),
            available: parseInt(available, 10),
            pricePerHour: parseFloat(pricePerHour),
            createdAt: new Date(),
            updatedAt: new Date(),
            currentUsers: currentUsers || {} // Agregar currentUsers si existe
        });

        const savedParking = await newParking.save(); // Guardar el documento en la base de datos
        res.status(201).json(savedParking);
    } catch (error) {
        res.status(500).json({ message: 'Error al agregar el estacionamiento', error });
    }
});


app.get('/api/reservations', async (req, res) => {
    try {
        const reservations = await Reservation.find().populate('userId', 'name'); // Esto poblara el campo 'userId' con solo el 'name'
        res.json(reservations);
    } catch (error) {
        res.status(500).json({ message: 'Error al obtener el historial de ocupación', error });
    }
});

app.post('/api/reservations', async (req, res) => {
    try {
        const { userId, parkingId, startTime, status, totalCost } = req.body;

        // Crear un nuevo objeto de reservación sin `endTime`
        const newReservation = new Reservation({
            userId,
            parkingId,
            startTime,
            status,
            totalCost
        });

        const savedReservation = await newReservation.save();
        res.status(201).json(savedReservation);
    } catch (error) {
        console.error('Error al crear la reservación:', error);
        res.status(500).json({ message: 'Error al crear la reservación', error });
    }
});


app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;
    try {
        const user = await User.findOne({ email });
        if (!user) {
            return res.status(401).json({ message: 'Usuario no encontrado' });
        }

        if (password !== user.password) {
            return res.status(401).json({ message: 'Contraseña incorrecta' });
        }

        res.json({ userId: user._id, name: user.name, role: user.role });
    } catch (error) {
        res.status(500).json({ message: 'Error al iniciar sesión', error });
    }
});

// Rutas para obtener y crear usuarios
app.get('/api/users', async (req, res) => {
    try {
        const users = await User.find(); // Obtener todos los usuarios
        res.json(users);
    } catch (error) {
        res.status(500).json({ message: 'Error al obtener los usuarios', error });
    }
});

app.post('/api/users', async (req, res) => {
    try {
        const { name, email, role, password } = req.body;

        // Crear un nuevo usuario
        const newUser = new User({
            name,
            email,
            role,
            password, // Asegúrate de manejar el almacenamiento seguro de contraseñas
            createdAt: new Date(),
            updatedAt: new Date()
        });

        const savedUser = await newUser.save(); // Guardar el usuario en la base de datos
        res.status(201).json(savedUser);
    } catch (error) {
        res.status(500).json({ message: 'Error al crear el usuario', error });
    }
});

app.get('/', (req, res) => {
    res.send('El servidor está funcionando correctamente.');
});

app.listen(port, () => {
    console.log(`Servidor escuchando en http://localhost:${port}`);
});
