const express = require('express');
const axios = require('axios');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(cors());

app.use(express.static(path.join(__dirname, 'public')));

app.get('/suggest', async (req, res) => {
    const userId = req.query.user_id || 0;
    const algo = req.query.algo || 'user';
    
    try {
        const pythonUrl = `http://127.0.0.1:5000/api/recommend?user_id=${userId}&algo=${algo}`;
        const aiResponse = await axios.get(pythonUrl);
        const movieIds = aiResponse.data.recommendations;

        const moviesWithDetails = movieIds.map(id => ({
            id: id,
            title: `Phim Bí Ẩn Số ${id}`,
            poster: `https://picsum.photos/seed/movie-${id}/300/450`,
            genre: algo === 'user' ? "Gợi ý từ Hàng xóm" : "Gợi ý từ Phim tương đồng"
        }));

        res.json({
            user_id: userId,
            algorithm: algo,
            data: moviesWithDetails
        });

    } catch (error) {
        res.status(500).json({ error: "Lỗi kết nối Server AI" });
    }
});

app.listen(3000, () => {
    console.log('http://localhost:3000');
});